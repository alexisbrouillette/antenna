#!/bin/bash
# ==============================================================================
# SLURM Batch Script: Fine-Tune MobileSAM Mask Decoder on GPU Cluster
# ==============================================================================
#SBATCH --job-name=sam_insects
#SBATCH --output=sam_train_%j.log
#SBATCH --error=sam_train_%j.err
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G

echo "=========================================================="
echo "Starting MobileSAM Fine-Tuning on Insect Anatomy"
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "CUDA Device: $CUDA_VISIBLE_DEVICES"
echo "=========================================================="

# 1. Load modules if on standard HPC cluster (e.g. Alliance / Compute Canada)
if command -v module &> /dev/null; then
    module load python/3.10 cuda cudnn 2>/dev/null || true
fi

# 2. Virtual environment setup
if [ -z "$VIRTUAL_ENV" ]; then
    if [ ! -d "venv" ]; then
        echo "Creating virtual environment..."
        python3 -m venv venv
    fi
    source venv/bin/activate
fi

# 3. Check dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Run MobileSAM Fine-Tuning (Freezes 6M image encoder, trains 4M mask decoder with Dice loss)
echo "Launching finetune_sam.py..."
python finetune_sam.py --epochs 30 --batch-size 4 --lr 0.0001 --device 0

echo "=========================================================="
echo "Fine-tuning completed at $(date)"
echo "Saved weights: weights/mobile_sam_finetuned.pt"
echo "=========================================================="

