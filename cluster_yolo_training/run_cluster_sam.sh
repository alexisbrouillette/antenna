#!/bin/bash
# ==============================================================================
# SLURM Cluster Batch Submission Script - MobileSAM Fine-Tuning
# ==============================================================================
#SBATCH --job-name=sam_finetune
#SBATCH --account=def-dgravel
#SBATCH --output=sam_train_%j.log
#SBATCH --error=sam_train_%j.err
#SBATCH --time=3:0:0
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gpus=nvidia_h100_80gb_hbm3_2g.20gb:1
#SBATCH --mem=128G

echo "=========================================================="
echo "Starting MobileSAM Fine-Tuning on GPU Cluster"
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

# 3. Run MobileSAM Fine-Tuning (Freezes 6M image encoder, trains 4M mask decoder with Dice loss)
echo "Launching finetune_sam.py..."
cd "$SCRIPT_DIR"
python finetune_sam.py --epochs 30 --batch-size 4 --lr 0.0001 --device 0

echo "=========================================================="
echo "Fine-tuning completed at $(date)"
echo "Saved weights: weights/mobile_sam_finetuned.pt"
echo "=========================================================="
