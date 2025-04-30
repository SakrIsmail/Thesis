#!/bin/bash
#SBATCH --job-name=faster_rcnn
#SBATCH --output=outputs/output_faster_rcnn.log
#SBATCH --error=outputs/error_faster_rcnn.log
#SBATCH --time=00:15:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1

# Load necessary modules
#module load anaconda3

eval "$(conda shell.bash hook)"

# Activate your conda environment
conda activate /var/scratch/sismail/my_env

# Run your Python script
python python/faster_rcnn.py

conda deactivate
