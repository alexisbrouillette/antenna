#!/bin/bash
# ==============================================================================
# SLURM Cluster Batch Submission Script (Compute Canada / Slurm / Any GPU cluster)
# ==============================================================================
#SBATCH --job-name=yolo_insects
#SBATCH --output=yolo_train_%j.log
#SBATCH --error=yolo_train_%j.err
#SBATCH --time=01:00:00
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

# 1. Load modules if on standard HPC cluster (e.g. Compute Canada / Alliance)
if command -v module &> /dev/null; then
    module load python/3.10 cuda cudnn 2>/dev/null || true
fi

# 2. Virtual environment setup (creates one if not already active)
if [ -z "$VIRTUAL_ENV" ]; then
    if [ ! -d "venv" ]; then
        echo "Creating virtual environment..."
        python3 -m venv venv
    fi
    source venv/bin/activate
fi

# 3. Install required packages
echo "Checking requirements..."
pip install --upgrade pip
pip install -r requirements.txt

# 4. Run Training
echo "Starting training..."
python train.py --epochs 80 --batch 16 --device 0

echo "=========================================================="
echo "Training completed at $(date)"
echo "Trained weights are saved in: weights/best.pt"
echo "=========================================================="

