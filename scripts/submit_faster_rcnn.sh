#!/bin/bash
#SBATCH --job-name=faster_rcnn
#SBATCH --output=output.log
#SBATCH --error=error.log
#SBATCH --time=00:15:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1

# Load necessary modules
module load anaconda3

# Activate your conda environment
source activate thesis

# Navigate to the directory containing your script
cd python

# Run your Python script
python faster_rcnn.py
