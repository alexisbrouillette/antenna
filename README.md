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
To run the automated consolidation and crop-downloader:
1. Ensure the raw CSV exports are located in `data/antenna_raw_data/`.
2. Launch Jupyter Lab or Notebook:
   ```bash
   jupyter lab
   ```
3. Open `notebooks/data_exploration.ipynb` and run the cells. This will merge the CSVs, filter for verified specimens, and download all target insect images directly to `data/antenna_field_images/`.
   * *Note: The notebook script automatically checks for existing files and skips duplicates, so re-running it will not overwrite your work.*

### 3. Load & Share Annotations
We use **X-AnyLabeling** to draw contours and label species. 

To run it and load annotations:
1. Launch the tool from your terminal (make sure the virtual environment is activated):
   ```bash
   xanylabeling
   ```
2. **Open Images Directory:** Click **Open Folder** in the GUI and select the `data/antenna_field_images/` directory.
3. The tool will automatically pair the images with their sidecar `.json` files and display the existing polygons.


---

## 🔄 Pipeline Phases

1. **Phase 1: Validation Sandbox Calibration**
   * Pre-flight checks and metric dimension calibration using the *InsectMorphoAI* reference dataset ($100$ specimens with paired dry-weight values).
2. **Phase 2: Silhouette Segmentation & Masking**
   * Isolate the outer body contours from the background trapping sheet using thresholding and morphological operations.
3. **Phase 3: Allometric Modeling**
   * Fit power-law scaling parameters ($M = a \cdot A^b$) to formulate regression relationships between pixel surface area and biomass.
