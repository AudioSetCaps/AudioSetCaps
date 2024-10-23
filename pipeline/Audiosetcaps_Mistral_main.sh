#!/bin/bash

echo "Activating clap environment..."
source ~/conda.sh
conda activate clap
python Audiosetcaps_Mistral_main.py

echo "Deactivating clap environment..."
conda deactivate

