#!/bin/bash

# Create output directory
mkdir -p outputs/gsm8k

# Install required packages if not already installed
pip install datasets

# Run GSM8K evaluation with a small sample size first
python3 gsm8k_evaluation.py --config config/gsm8k_config.yaml --model "gpt2-medium" --num_samples 5 --output_dir outputs/gsm8k
