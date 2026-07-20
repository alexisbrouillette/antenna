# Insect Biomass & Trait Extraction Pipeline Tracker

This file serves as the master tracking document for our 2D Computer Vision (Area-to-Biomass) pipeline. It documents environmental settings, dataset verification, image processing metrics, and final model validation scores.

---

## 🛠 Project Environment & Setup
* **Virtual Environment Name:** `env`
* **Jupyter Kernel:** `Python (env)`
* **Key Dependencies:**
  * `opencv-python` (Image load and preprocessing)
  * `pandas` & `openpyxl` (Tabular metadata manipulation)
  * `scikit-image` & `numpy` (Advanced contour segmentation & pixel calculation)
  * `matplotlib` (Visualization checks)
  * `scikit-learn` (Allometric regression modeling)

---

## 📊 Phase 1: Validation Sandbox Calibration
We are using the **InsectMorphoAI Biomass Validation Dataset** (100 Tachinidae specimens with paired high-resolution images and precise microbalance dry weights) to test and validate our physical dimension measurements and allometric modeling before applying them to our Nunavik moth traps.

### 1. Dataset Verification Log
* **Metadata Sheet:** `Validation_Measurements` inside `Supplementary_Data.xlsx`
* **Total Records:** 100 specimens
* **Images Directory:** `../data/Validation_Images`
* **Sample Test Check:**
  * **Filename:** `CAS001_CAS0000002_stacked_02_H.png`
  * **Species:** *Dexia rustica*
  * **Image Resolution:** $4032 	imes 3040$ pixels
  * **Ground Truth Dry Weight ($M$):** $7.20 \text{ mg}$ ($0.0072 \text{ g}$ in Excel)

---

## 🚀 Pipeline Development Log

### [Completed] Step 1: Directory Setup & Target Database Mapping
* Verified data paths relative to the notebook directory (`../data/`).
* Confirmed connection between spreadsheet `Filename` keys and raw image files in directory.
* Successfully imported all required modules without errors.

### [Pending] Step 2: Adaptive 2D Foreground Segmentation & Masking
* **Objective:** Isolate the insect's outer silhouette boundary from the background sheet.
* **Techniques to Test:**
  * Contrast-limited adaptive histogram equalization (CLAHE).
  * Otsu's Global Thresholding vs. Local Adaptive Thresholding.
  * Morphological clean-up (dilation/erosion to close internal body holes).

### [Pending] Step 3: Metric Conversions & Spatial Calibration
* **Objective:** Find the spatial scale factor ($S = \text{pixels per mm}$) and calculate physical length, width, and surface area ($A$) in physical units.
* **Math Equations:**
  $$L_{\text{mm}} = \frac{L_{\text{pixels}}}{S}$$
  $$A_{\text{mm}^2} = \frac{A_{\text{pixels}}}{S^2}$$

### [Pending] Step 4: Allometric Modeling & Validation Curve
* **Objective:** Fit power-law scaling models to predict biomass ($M$) from area ($A$) and analyze error metrics.
* **Model Parameters:**
  $$M = a \cdot A^b$$
* **Key Metrics to Track:**
  * Coefficient of Determination ($R^2$)
  * Mean Absolute Percentage Error (MAPE)
