#!/bin/bash

# Create output directory
mkdir -p outputs/test

# Run HATO test with sample dataset
python3 test_hato.py --config config/test_config.yaml --model_size 0.6b --sample_size 5 --output_dir outputs/test
