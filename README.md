# Insect Biomass & Trait Extraction Pipeline

This repository contains the pipeline and toolset for the 2D Computer Vision (Area-to-Biomass) research project. The goal is to estimate insect biomass ($M$) from physical dimensions ($L$, $W$) and contour-segmented surface area ($A$) using allometric scaling models ($M = a \cdot A^b$).

---

## 📁 Repository Structure

Ensure the directory structure matches the following tree:

```text
antenna/
├── README.md                           # This instructions document
├── requirements.txt                    # Project dependencies
├── pipeline_tracker.md                 # Status and validation log
├── data_exploration.py                 # Command-line downloader script (Recommended)
├── notebooks/
│   └── data_exploration.ipynb          # Dataset consolidation and download notebook
└── data/
    ├── antenna_raw_data/               # [Place raw data CSV exports here]
    │   ├── barro-colorado-island-research-station_export-97.csv
    │   ├── insectarium-de-montreal_export-103.csv
    │   └── insectarium-de-montreal_export-72.csv
    ├── antenna_field_images/           # [Stores downloaded crop images & sidecar JSON annotations]
    │   ├── ind_10072.jpg
    │   ├── ind_10072.json
    │   └── ...
    └── antenna_processed_dataset.csv   # Consolidated tabular metadata index

```

---

## 🛠️ Step-by-Step setup for Colleagues

To get set up with minimal friction and start segmenting immediately, follow these steps:

### 1. Set Up Environment & Install Dependencies
First, create and activate a Python virtual environment, then install the pinned project requirements (this will automatically clone and install **X-AnyLabeling**):

```bash
# Create a virtual environment
python3 -m venv env

# Activate it
source env/bin/activate

# Install the dependencies (including X-AnyLabeling from Git)
   
pip install -r requirements.txt
```

### 2. Download the Image Dataset
To run the automated consolidation and crop-downloader, ensure the raw CSV exports are located in `data/antenna_raw_data/`. 

Then, choose one of the following methods to download the images:

#### Option A: Run the Python Script (Recommended)
Simply run the standalone script directly from your terminal:
```bash
python data_exploration.py
```

#### Option B: Run the Jupyter Notebook (Alternative)
If you prefer using notebooks (e.g. in VS Code or JupyterLab):
1. Start Jupyter:
   ```bash
   jupyter lab
   ```
2. Open `notebooks/data_exploration.ipynb` and run all cells.

*Note: Both methods automatically check for existing files and skip duplicates, so running them again will not overwrite your work or redownload existing images.*


### 3. Load & Share Annotations
We use **X-AnyLabeling** to draw contours and label species. 

To run it and load annotations:
1. Launch the tool from your terminal (make sure the virtual environment is activated):
   ```bash
   xanylabeling
   ```
2. **Open Images Directory:** Click **Open Folder** in the GUI and select the `data/antenna_field_images/` directory.
3. The tool will automatically pair the images with their sidecar `.json` files and display the existing polygons.
4. Click on the `AI` icon on the left bar and select the model. I used `MobileSAM` since it works well on my laptop but any model will do fine for the preprocessing.

---

## ⚡ Useful Commands Cheat Sheet

Here are the primary commands for the active annotation, sync, dataset reconstruction, and training workflow:

### 1. Re-annotate Active Set with AI
Runs the trained YOLO-seg model to generate or update polygon masks for unannotated images in AnyLabeling. **Your human-verified annotations and manual edits are 100% protected and will never be overwritten.**
```bash
# Re-predict all images in the active backlog using the latest weights
python cluster_yolo_training/predict_and_annotate.py --overwrite-ai

# Re-predict with MobileSAM refinement (sharp promptable mask boundaries)
python cluster_yolo_training/predict_and_annotate.py --use-sam --overwrite-ai

# Re-predict only the Field Camera Trap track or iNaturalist track
python cluster_yolo_training/predict_and_annotate.py --track field --overwrite-ai
python cluster_yolo_training/predict_and_annotate.py --track inat --overwrite-ai

# Re-predict a single species (for quick checks)
python cluster_yolo_training/predict_and_annotate.py --species "Eacles_imperialis" --overwrite-ai
```

### 1b. Benchmark: Pure YOLO vs. YOLO + SAM
Compare segmentation accuracy and generate side-by-side 4-panel visual comparison figures (`[Original]` | `[Ground Truth]` | `[Pure YOLO]` | `[YOLO + SAM]`):
```bash
python cluster_yolo_training/compare_yolo_sam.py --num-samples 10
```

### 2. Annotation Backlog & Triage
```bash
# Check status & sync all verified annotations from active set to master storage
python annotation_backlog.py sync

# Triage & rank images (Top 40 high-confidence in clean_to_approve/, Bottom 20 edge cases in hand_annotate/)
python annotation_backlog.py triage

# Evacuate verified images to master storage and refill active slots with fresh candidates
python annotation_backlog.py replenish

# Approve all clean candidates for a species at once
python annotation_backlog.py approve "Eacles imperialis"
```

### 3. Rebuild Dataset & Train
```bash
# 1. Rebuild the YOLO/SAM dataset from all verified annotations (field + degraded iNat)
python cluster_yolo_training/build_dataset.py

# 2. Train YOLO11s-seg on GPU cluster
cd cluster_yolo_training
sbatch run_cluster.sh
# Or interactive GPU: python train.py --model yolo11s-seg.pt --epochs 100 --batch 16 --device 0

# 3. Fine-Tune MobileSAM Mask Decoder (3-5 min on GPU)
sbatch run_cluster_sam.sh
# Or interactive GPU: python finetune_sam.py --epochs 30 --batch-size 4 --lr 0.0001 --device 0
```


