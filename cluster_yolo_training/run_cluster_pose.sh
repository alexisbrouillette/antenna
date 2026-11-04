#!/bin/bash
# ==============================================================================
# SLURM Cluster Batch Submission Script - Insect YOLO-Pose
# ==============================================================================
#SBATCH --job-name=yolo_pose
#SBATCH --account=def-dgravel
#SBATCH --output=pose_train_%j.log
#SBATCH --error=pose_train_%j.err
#SBATCH --time=3:0:0
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gpus=nvidia_h100_80gb_hbm3_2g.20gb:1
#SBATCH --mem=128G

echo "=========================================================="
echo "Starting YOLO-Pose Insect Skeleton Training on GPU Cluster"
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "CUDA Devices: $CUDA_VISIBLE_DEVICES"
echo "=========================================================="

if command -v module &> /dev/null; then
    module load opencv cuda cudnn 2>/dev/null || true
fi

source /home/alexisb1/antenna/env/bin/activate
cd /home/alexisb1/antenna/cluster_yolo_training

echo "Launching train_pose.py in $(pwd)..."
python train_pose.py --model yolo11s-pose.pt --epochs 120 --batch 16 --device 0

echo "=========================================================="
echo "Pose training completed at $(date)"
echo "Saved weights: weights/best_pose.pt"
echo "=========================================================="
