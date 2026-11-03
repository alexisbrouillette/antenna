#!/bin/bash
# ==============================================================================
# SLURM Cluster Batch Submission Script (Compute Canada / Slurm / fir)
# ==============================================================================
#SBATCH --job-name=yolo_insects
#SBATCH --account=def-dgravel_gpu
#SBATCH --output=yolo_train_%j.log
#SBATCH --error=yolo_train_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G

echo "=========================================================="
echo "Starting YOLO-Seg Insect Anatomy Training on GPU Cluster"
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "CUDA Devices: $CUDA_VISIBLE_DEVICES"
echo "=========================================================="

# 1. Load modules on Compute Canada / Alliance
if command -v module &> /dev/null; then
    module load opencv cuda cudnn 2>/dev/null || true
fi

# 2. Virtual environment setup
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$SCRIPT_DIR/../env/bin/activate" ]; then
    source "$SCRIPT_DIR/../env/bin/activate"
elif [ -f "$SCRIPT_DIR/venv/bin/activate" ]; then
    source "$SCRIPT_DIR/venv/bin/activate"
fi

# 3. Run Training (11.2M Small model, 100 epochs)
echo "Starting training..."
cd "$SCRIPT_DIR"
python train.py --model yolo11s-seg.pt --epochs 100 --batch 16 --device 0

echo "=========================================================="
echo "Training completed at $(date)"
echo "Trained weights are saved in: weights/best.pt"
echo "=========================================================="
