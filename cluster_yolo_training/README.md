# 🦋 Insect Anatomy YOLO-Seg GPU Training Package

Everything you need to train a custom YOLO instance segmentation model on your insect anatomical annotations (`wing`, `leg`, `antenna`, `head`, `body`, `full-body`) and auto-generate masks for AnyLabeling.

---

## 📁 Package Contents

```
cluster_yolo_training/
├── dataset/
│   ├── images/train/ (194 field camera trap images)
│   ├── images/val/   (34 validation images)
│   ├── labels/train/ (YOLO normalized polygon .txt)
│   ├── labels/val/   (YOLO normalized polygon .txt)
│   └── dataset.yaml  (Classes: 0:full-body, 1:wing, 2:leg, 3:antenna, 4:head, 5:body)
├── train.py                  # Training script (supports GPU or CPU)
├── run_cluster.sh            # Slurm batch script for GPU clusters
├── predict_and_annotate.py   # Pre-annotation script (creates LabelMe JSONs)
├── requirements.txt          # Dependencies (ultralytics, torch, etc.)
└── README.md
```

**Total package size:** ~10 MB (easy to copy or scp in seconds).

---

## 🚀 Step 1: Copy to Your GPU Cluster

From your local computer terminal:

```bash
# Example with scp:
scp -r cluster_yolo_training/ username@cluster.computecanada.ca:~/

# Or compress first:
tar -czvf cluster_yolo.tar.gz cluster_yolo_training/
scp cluster_yolo.tar.gz username@cluster.computecanada.ca:~/
```

---

## ⚡ Step 2: Run Training on the Cluster

### Option A: Train YOLO11s-seg (Anatomical Detection + Segmentation)
```bash
# If using Slurm:
sbatch run_cluster.sh

# Or run interactively in a GPU session (10.1M YOLO11 Small model):
python train.py --model yolo11s-seg.pt --epochs 100 --batch 16 --device 0
```
- **Training time on a GPU (e.g. V100, A100, T4):** ~**5 to 10 minutes**.
- **Output:** `weights/best.pt`

### Option B: Fine-Tune MobileSAM Mask Decoder (Boundary Refinement)
Freezes the 6.0M image encoder and trains the 4.0M mask decoder with Dice loss on insect masks:
```bash
# If using Slurm:
sbatch run_cluster_sam.sh

# Or run interactively in a GPU session:
python finetune_sam.py --epochs 30 --batch-size 4 --lr 0.0001 --device 0
```
- **Training time on a GPU:** ~**3 to 5 minutes**.
- **Output:** `weights/mobile_sam_finetuned.pt`

---

## 🔬 Benchmark: Pure YOLO vs. YOLO + SAM
Compare segmentation accuracy across validation samples:
```bash
python compare_yolo_sam.py --num-samples 10 --sam weights/mobile_sam_finetuned.pt
```

---

## 📥 Step 3: Copy Back & Pre-Annotate All Images

1. Copy `best.pt` back to your local project:
   ```bash
   scp username@cluster:~/cluster_yolo_training/runs/insect_anatomy_v2/weights/best.pt cluster_yolo_training/weights/best.pt
   ```

2. Run the pre-annotator on your active unannotated images:
   ```bash
   # Re-predict and refresh AI masks on the active set with the latest model
   python cluster_yolo_training/predict_and_annotate.py --overwrite-ai

   # Or filter by track or species:
   python cluster_yolo_training/predict_and_annotate.py --track field --overwrite-ai
   python cluster_yolo_training/predict_and_annotate.py --species "Eacles_imperialis" --overwrite-ai
   ```

3. **Open AnyLabeling:**
   - Open any folder in `data/active_annotation_set/field` or `data/active_annotation_set/inat`.
   - **All polygon masks are already generated with confidence scores and quality ratings!**
   - You just do quick quality checks: delete any unwanted contour, nudge points, and save (`Ctrl+S`). Manual edits and approvals are automatically protected against overwriting.

---

## 🛠️ (Optional) Test Locally on CPU

If you want to test the script right now without a GPU:
```bash
./env/bin/pip install ultralytics
python cluster_yolo_training/train.py --epochs 2 --batch 4 --device cpu
```

