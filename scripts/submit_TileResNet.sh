#!/bin/bash
#SBATCH --job-name=TileResNet
#SBATCH --output=outputs/output_TileResNet
#SBATCH --error=outputs/error_TileResNet
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --gres=gpu:1


module load cuda12.3/toolkit/12.3
module load cuDNN/cuda12.3/9.1.0.70

eval "$(conda shell.bash hook)"

# Activate your conda environment
conda activate /var/scratch/sismail/my_env

# Run your Python script
python python/MobileTileNet/TileResNet.py

conda deactivate
