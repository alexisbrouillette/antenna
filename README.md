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

## 🎯 Targeted Annotation Backlog Manager

To avoid loading thousands of unannotated images in X-AnyLabeling, we use `annotation_backlog.py`. This script dynamically manages an active subset of images to annotate. It targets species that have **fewer than 5 annotations**, and exposes up to 8 candidate images per species (giving you options if some images have poor quality). As soon as a species reaches 5 annotations, its unannotated candidates are automatically removed from the backlog.

### How to use it:

1. **Populate the Backlog Workspace:**
   Run the backlog script to automatically build a curated workspace in `data/active_annotation_set/`:
   ```bash
   python annotation_backlog.py
   ```
2. **Open in X-AnyLabeling:**
   Launch the tool and open the folder `data/active_annotation_set/`. You will see all your existing annotations and a selection of candidate images for species that still need coverage.
3. **Sync Your Progress Back:**
   After you finish an annotation session, run the sync command to push all newly created `.json` files back into your master dataset directory (`data/antenna_field_images/`):
   ```bash
   python annotation_backlog.py sync
   ```

