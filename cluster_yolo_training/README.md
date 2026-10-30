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

SSH into your cluster:

```bash
cd cluster_yolo_training

# If using Slurm:
sbatch run_cluster.sh

# Or run interactively in a GPU session:
pip install -r requirements.txt
python train.py --epochs 80 --batch 16 --device 0
```

- **Training time on a GPU (e.g. V100, A100, T4):** ~**3 to 5 minutes**.
- **Output:** When finished, your model weights will be saved at:
  - `weights/best.pt`
  - `weights/best.onnx`

---

## 📥 Step 3: Copy Back & Pre-Annotate All Images

1. Copy `weights/best.pt` back to your local project:
   ```bash
   scp username@cluster:~/cluster_yolo_training/weights/best.pt cluster_yolo_training/weights/best.pt
   ```

2. Run the pre-annotator on your active unannotated images:
   ```bash
   python cluster_yolo_training/predict_and_annotate.py --target data/active_annotation_set
   ```

3. **Open AnyLabeling:**
   - Open any folder in `data/active_annotation_set/field` or `data/active_annotation_set/inat`.
   - **All polygon masks are already there on every image!**
   - You just do quick quality checks: delete any unwanted contour, nudge points, and save (`Ctrl+S`).

---

## 🛠️ (Optional) Test Locally on CPU

If you want to test the script right now without a GPU:
```bash
./env/bin/pip install ultralytics
python cluster_yolo_training/train.py --epochs 2 --batch 4 --device cpu
```

