#!/bin/bash
#SBATCH --job-name=graph_rcnn_MobileNet_baseline_v2
#SBATCH --output=outputs/graph_rcnn/output_graph_rcnn_MobileNet_baseline_v2
#SBATCH --error=outputs/graph_rcnn/error_graph_rcnn_MobileNet_baseline_v2
#SBATCH --time=16:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --exclusive


module load cuda12.3/toolkit/12.3
module load cuDNN/cuda12.3/9.1.0.70

eval "$(conda shell.bash hook)"

# Activate your conda environment
conda activate /var/scratch/$USER/my_env

# Run your Python script
python python/graph_rcnn/graph_rcnn_MobileNet_baseline_v2.py 

conda deactivate
