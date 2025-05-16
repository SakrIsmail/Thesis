#!/bin/bash
#SBATCH --job-name=yolo_baseline
#SBATCH --output=outputs/yolo/output_yolo_baseline
#SBATCH --error=outputs/yolo/error_yolo_baseline
#SBATCH --time=06:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1


module load cuda12.3/toolkit/12.3
module load cuDNN/cuda12.3/9.1.0.70

eval "$(conda shell.bash hook)"

# Activate your conda environment
conda activate /var/scratch/sismail/my_env

# Run your Python script
python python/yolo/yolo_baseline.py 

conda deactivate
