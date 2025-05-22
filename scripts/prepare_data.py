import os
import json
import requests
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from typing import Dict, List, Optional, Union
import logging
import hashlib
from concurrent.futures import ThreadPoolExecutor
import gzip
import shutil

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatasetDownloader:
    """Download and prepare math reasoning datasets"""

    DATASETS = {
        "gsm8k": {
            "train_url": "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/train.jsonl",
            "test_url": "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl",
            "format": "jsonl",
        },
        "math": {
            "train_url": "https://raw.githubusercontent.com/hendrycks/math/master/data/train.jsonl.gz",
            "test_url": "https://raw.githubusercontent.com/hendrycks/math/master/data/test.jsonl.gz",
            "format": "jsonl_gz",
        },
        "aqua": {
            "train_url": "https://raw.githubusercontent.com/deepmind/AQuA/master/train.json",
            "test_url": "https://raw.githubusercontent.com/deepmind/AQuA/master/test.json",
            "format": "json",
        },
    }

    def __init__(self, data_dir: str = "data"):
        """
        Initialize dataset downloader

        Args:
            data_dir: Directory to save datasets
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories for each dataset
        for dataset in self.DATASETS:
            (self.data_dir / dataset).mkdir(exist_ok=True)

    def _download_file(self, url: str, output_path: Path) -> bool:
        """
        Download a file with progress bar

        Args:
            url: URL to download from
            output_path: Path to save file

        Returns:
            bool: Whether download was successful
        """
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()

            total_size = int(response.headers.get("content-length", 0))
            block_size = 8192

            with (
                open(output_path, "wb") as f,
                tqdm(
                    desc=output_path.name, total=total_size, unit="iB", unit_scale=True
                ) as pbar,
            ):
                for data in response.iter_content(block_size):
                    size = f.write(data)
                    pbar.update(size)
            return True

        except Exception as e:
            logger.error(f"Error downloading {url}: {e}")
            return False

    def _extract_gz(self, gz_path: Path, output_path: Path) -> bool:
        """
        Extract gzipped file

        Args:
            gz_path: Path to gzipped file
            output_path: Path to save extracted file

        Returns:
            bool: Whether extraction was successful
        """
        try:
            with gzip.open(gz_path, "rb") as f_in:
                with open(output_path, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)
            return True
        except Exception as e:
            logger.error(f"Error extracting {gz_path}: {e}")
            return False

    def _process_gsm8k(self, data: List[Dict]) -> List[Dict]:
        """Process GSM8K data format"""
        processed = []
        for item in data:
            processed.append(
                {
                    "problem": item["question"],
                    "solution": item["answer"],
                    "answer": item["answer"].split("####")[-1].strip(),
                }
            )
        return processed

    def _process_math(self, data: List[Dict]) -> List[Dict]:
        """Process MATH dataset format"""
        processed = []
        for item in data:
            processed.append(
                {
                    "problem": item["problem"],
                    "solution": item["solution"],
                    "answer": item["answer"],
                    "level": item["level"],
                    "type": item["type"],
                }
            )
        return processed

    def _process_aqua(self, data: List[Dict]) -> List[Dict]:
        """Process AQUA dataset format"""
        processed = []
        for item in data:
            processed.append(
                {
                    "problem": item["question"],
                    "options": item["options"],
                    "solution": item["rationale"],
                    "answer": item["correct"],
                }
            )
        return processed

    def download_dataset(self, dataset_name: str, force: bool = False) -> bool:
        """
        Download and prepare a dataset

        Args:
            dataset_name: Name of dataset to download
            force: Whether to force redownload

        Returns:
            bool: Whether download was successful
        """
        if dataset_name not in self.DATASETS:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        dataset_info = self.DATASETS[dataset_name]
        dataset_dir = self.data_dir / dataset_name

        # Download train and test files
        for split in ["train", "test"]:
            url = dataset_info[f"{split}_url"]
            output_path = dataset_dir / f"{split}.{dataset_info['format']}"

            # Skip if file exists and not forcing
            if output_path.exists() and not force:
                logger.info(f"{split} file already exists: {output_path}")
                continue

            # Download file
            logger.info(f"Downloading {split} file from {url}")
            if not self._download_file(url, output_path):
                return False

            # Extract if gzipped
            if dataset_info["format"] == "jsonl_gz":
                extracted_path = dataset_dir / f"{split}.jsonl"
                if not self._extract_gz(output_path, extracted_path):
                    return False
                output_path = extracted_path

        # Process and save in standard format
        for split in ["train", "test"]:
            input_path = dataset_dir / f"{split}.jsonl"
            if dataset_info["format"] == "json":
                input_path = dataset_dir / f"{split}.json"

            # Read data
            if dataset_info["format"] == "json":
                with open(input_path) as f:
                    data = json.load(f)
            else:
                data = []
                with open(input_path) as f:
                    for line in f:
                        data.append(json.loads(line))

            # Process data
            if dataset_name == "gsm8k":
                processed = self._process_gsm8k(data)
            elif dataset_name == "math":
                processed = self._process_math(data)
            elif dataset_name == "aqua":
                processed = self._process_aqua(data)

            # Save processed data
            output_path = dataset_dir / f"{split}_processed.jsonl"
            with open(output_path, "w") as f:
                for item in processed:
                    f.write(json.dumps(item) + "\n")

            logger.info(f"Saved processed {split} data to {output_path}")

        return True

    def download_all(self, force: bool = False) -> Dict[str, bool]:
        """
        Download all datasets

        Args:
            force: Whether to force redownload

        Returns:
            Dict mapping dataset names to success status
        """
        results = {}
        for dataset in self.DATASETS:
            logger.info(f"Downloading dataset: {dataset}")
            results[dataset] = self.download_dataset(dataset, force)
        return results


def main():
    """Main function to download datasets"""
    import argparse

    parser = argparse.ArgumentParser(description="Download math reasoning datasets")
    parser.add_argument("--data_dir", default="data", help="Directory to save datasets")
    parser.add_argument(
        "--dataset",
        choices=["gsm8k", "math", "aqua", "all"],
        default="all",
        help="Dataset to download",
    )
    parser.add_argument("--force", action="store_true", help="Force redownload")

    args = parser.parse_args()

    downloader = DatasetDownloader(data_dir=args.data_dir)

    if args.dataset == "all":
        results = downloader.download_all(force=args.force)
        for dataset, success in results.items():
            logger.info(f"{dataset}: {'Success' if success else 'Failed'}")
    else:
        success = downloader.download_dataset(args.dataset, force=args.force)
        logger.info(f"{args.dataset}: {'Success' if success else 'Failed'}")


if __name__ == "__main__":
    main()
