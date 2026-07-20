import os
import glob
import urllib.request
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# --- CONFIGURATION ---
# Path settings relative to the repository root
DATA_DIR = "data/antenna_raw_data"
PARENT_DATA_DIR = "data"
IMAGES_OUT_DIR = os.path.join(PARENT_DATA_DIR, "antenna_field_images")
CONSOLIDATED_CSV = os.path.join(PARENT_DATA_DIR, "antenna_processed_dataset.csv")

# Create output directories if they don't exist
os.makedirs(IMAGES_OUT_DIR, exist_ok=True)

# 1. Dynamically scan the folder for ALL CSV files
csv_pattern = os.path.join(DATA_DIR, "*.csv")
csv_files = glob.glob(csv_pattern)

# Prevent loading our own output tracking CSV if it's placed in the same folder
csv_files = [f for f in csv_files if os.path.abspath(f) != os.path.abspath(CONSOLIDATED_CSV)]

print(f"🔍 Scanning directory: {DATA_DIR}")
if not csv_files:
    raise FileNotFoundError(f"❌ No CSV files found in {DATA_DIR}. Please check the folder path!")
else:
    print(f"✔ Found {len(csv_files)} CSV files to process:")
    for f in csv_files:
        print(f"  - {os.path.basename(f)}")

# 2. Combine and read all discovered CSV files
df_list = []
for file in csv_files:
    try:
        df_temp = pd.read_csv(file, encoding="utf-8", encoding_errors="ignore")
        df_list.append(df_temp)
    except Exception as e:
        print(f"❌ Error reading {os.path.basename(file)}: {e}")

if not df_list:
    raise ValueError("No data could be loaded from the CSV files.")

df_all = pd.concat(df_list, ignore_index=True)
print(f"\nMerged database shape: {len(df_all)} total raw rows across all files.")

# 3. Filter Records (Keep high-confidence biological observations that are verified)
exclude_classes = ["Not Lepidoptera", "Not Identifiable", "Unknown", "Not_Lepidoptera"]
df_filtered = df_all[
    (df_all["verification_status"].isin([True, "Verified"])) & 
    (~df_all["determination_name"].isin(exclude_classes)) &
    (df_all["best_detection_url"].notna())
].copy()

print(f"Filtered down to {len(df_filtered)} high-confidence insect observations.")

# 4. Extract target columns
df_extracted = pd.DataFrame({
    'individual_id': df_filtered['id'],
    'species': df_filtered['determination_name'],
    'location': df_filtered['deployment_name'],
    'pixel_width': df_filtered['best_detection_width'],
    'pixel_height': df_filtered['best_detection_height'],
    'image_url': df_filtered['best_detection_url']
}).drop_duplicates(subset=['individual_id'])

# Create local relative file paths for each downloaded image
df_extracted['local_image_path'] = df_extracted.apply(
    lambda row: os.path.join("antenna_field_images", f"ind_{int(row['individual_id'])}.jpg"), axis=1
)

# 5. Downloader Function
def download_image(row):
    url = row['image_url']
    dest = os.path.join(PARENT_DATA_DIR, row['local_image_path'])
    
    # Avoid duplicate downloads if we re-run the script
    if os.path.exists(dest):
        return True
        
    try:
        # Simple User-Agent header so the cloud server allows the script to fetch the file
        req = urllib.request.Request(
            url, 
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        )
        with urllib.request.urlopen(req, timeout=10) as response, open(dest, 'wb') as out_file:
            out_file.write(response.read())
        return True
    except Exception as e:
        return f"Error downloading ID {row['individual_id']}: {e}"

# 6. Run Batch Download in Parallel Threads
print(f"\n🚀 Downloading {len(df_extracted)} insect crop images to '{IMAGES_OUT_DIR}'...")
records_dict = df_extracted.to_dict(orient='records')

results = []
with ThreadPoolExecutor(max_workers=8) as executor:
    results = list(tqdm(executor.map(download_image, records_dict), total=len(records_dict)))

failed_downloads = [r for r in results if r is not True]
print(f"\n✨ Download sequence finished.")
print(f"  - Successfully verified/downloaded: {len(records_dict) - len(failed_downloads)} crops")
if failed_downloads:
    print(f"  - Failed downloads: {len(failed_downloads)} (These URLs might be offline/slow)")

# 7. Save our clean database to disk
df_extracted.to_csv(CONSOLIDATED_CSV, index=False)
print(f"✔ Saved processed dataset tracking index to: {CONSOLIDATED_CSV}")
