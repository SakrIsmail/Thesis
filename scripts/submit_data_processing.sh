#!/bin/bash
#SBATCH --job-name=faster_rcnn_MobileNet_augmented
#SBATCH --output=outputs/output_data_processed
#SBATCH --error=outputs/output_data_processed
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G 
#SBATCH --time=04:00:00    



module load cuda12.3/toolkit/12.3
module load cuDNN/cuda12.3/9.1.0.70

eval "$(conda shell.bash hook)"

# Activate your conda environment
conda activate /var/scratch/sismail/my_env

# Run your Python script
python python/data_cleaning/data_processing_direct_missing.py

conda deactivate
