#!/bin/bash
#SBATCH --job-name=faster_rcnn_MobileNet_missing_baseline
#SBATCH --output=outputs/faster_rcnn/output_faster_rcnn_MobileNet_missing_baseline
#SBATCH --error=outputs/faster_rcnn/error_faster_rcnn_MobileNet_missing_baseline
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
python python/faster_rcnn/faster_rcnn_MobileNet_missing_baseline.py 

conda deactivate
