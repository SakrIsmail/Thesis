#!/bin/bash
#SBATCH --job-name=faster_rcnn_MobileNet_baseline_tuning
#SBATCH --output=outputs/faster_rcnn/output_faster_rcnn_MobileNet_baseline_tuning
#SBATCH --error=outputs/faster_rcnn/error_faster_rcnn_MobileNet_baseline_tuning
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
python python/faster_rcnn/faster_rcnn_MobileNet_baseline_tuning.py 

conda deactivate
