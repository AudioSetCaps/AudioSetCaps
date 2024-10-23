#!/bin/bash

echo "Activating qwen environment..."
source ~/conda.sh
conda activate qwen
python Audiosetcaps_qwen_main.py

echo "Deactivating qwen environment..."
conda deactivate


