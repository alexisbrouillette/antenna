import glob
import os
import socket
import urllib.request
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
from tqdm import tqdm

# --- CONFIGURATION ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DATA_DIR = os.path.join(SCRIPT_DIR, "data")
DATA_DIR = os.path.join(PARENT_DATA_DIR, "antenna_raw_data")
IMAGES_OUT_DIR = os.path.join(PARENT_DATA_DIR, "antenna_field_images")
CONSOLIDATED_CSV = os.path.join(
    PARENT_DATA_DIR, "antenna_processed_dataset.csv"
)

# Global socket timeout to prevent hung threads during stalled network reads
socket.setdefaulttimeout(15)

os.makedirs(IMAGES_OUT_DIR, exist_ok=True)

# 1. Scan folder for raw CSV files
csv_pattern = os.path.join(DATA_DIR, "*.csv")
csv_files = [
    f
    for f in glob.glob(csv_pattern)
    if os.path.abspath(f) != os.path.abspath(CONSOLIDATED_CSV)
]

print(f"🔍 Scanning directory: {DATA_DIR}")
if not csv_files:
    raise FileNotFoundError(
        f"❌ No CSV files found in {DATA_DIR}. Please check the folder path!"
    )

print(f"✔ Found {len(csv_files)} CSV files to process:")
for f in csv_files:
    print(f"  - {os.path.basename(f)}")

# 2. Combine all CSV files
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
print(
    f"\nMerged database shape: {len(df_all)} total raw rows across all files."
)

# 3. Filter Records
exclude_classes = [
    "Not Lepidoptera",
    "Not Identifiable",
    "Unknown",
    "Not_Lepidoptera",
]

# Ensure verification_status is compared safely across bool and string representations
valid_status = [True, "True", "true", "Verified", "verified"]

df_filtered = df_all[
    (df_all["verification_status"].isin(valid_status))
    & (~df_all["determination_name"].isin(exclude_classes))
    & (df_all["best_detection_url"].notna())
    & (df_all["id"].notna())
].copy()

# Enforce integer ID typing safely
df_filtered["id"] = pd.to_numeric(df_filtered["id"], errors="coerce")
df_filtered = df_filtered.dropna(subset=["id"])
df_filtered["id"] = df_filtered["id"].astype(int)

print(
    f"Filtered down to {len(df_filtered)} high-confidence insect observations."
)

# 4. Extract target columns
df_extracted = pd.DataFrame(
    {
        "individual_id": df_filtered["id"],
        "species": df_filtered["determination_name"].astype(str).str.strip(),
        "location": df_filtered["deployment_name"],
        "pixel_width": df_filtered["best_detection_width"],
        "pixel_height": df_filtered["best_detection_height"],
        "image_url": df_filtered["best_detection_url"],
    }
).drop_duplicates(subset=["individual_id"])

# Construct consistent image paths
df_extracted["local_image_path"] = df_extracted["individual_id"].apply(
    lambda x: os.path.join("antenna_field_images", f"ind_{x}.jpg")
)


# 5. Atomic Downloader Function
def download_image(row):
    url = row["image_url"]
    dest = os.path.join(PARENT_DATA_DIR, row["local_image_path"])
    temp_dest = f"{dest}.tmp"

    # Avoid duplicate downloads if file exists and has non-zero size
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return True

    try:
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            },
        )
        with (
            urllib.request.urlopen(req, timeout=12) as response,
            open(temp_dest, "wb") as out_file,
        ):
            out_file.write(response.read())

        # Atomically rename once download is complete
        os.replace(temp_dest, dest)
        return True
    except Exception as e:
        if os.path.exists(temp_dest):
            try:
                os.remove(temp_dest)
            except OSError:
                pass
        return f"Error downloading ID {row['individual_id']}: {e}"


# 6. Run Batch Download in Parallel Threads
print(
    f"\n🚀 Verifying/Downloading {len(df_extracted)} insect crop images to '{IMAGES_OUT_DIR}'..."
)
records_dict = df_extracted.to_dict(orient="records")

with ThreadPoolExecutor(max_workers=8) as executor:
    results = list(
        tqdm(
            executor.map(download_image, records_dict), total=len(records_dict)
        )
    )

failed_downloads = [r for r in results if r is not True]
print(f"\n✨ Download sequence finished.")
print(
    f"  - Successfully verified/downloaded: {len(records_dict) - len(failed_downloads)} crops"
)
if failed_downloads:
    print(
        f"  - Failed downloads: {len(failed_downloads)} (These URLs might be offline/slow)"
    )

# 7. Save clean tracking index to disk
df_extracted.to_csv(CONSOLIDATED_CSV, index=False)
print(f"✔ Saved processed dataset tracking index to: {CONSOLIDATED_CSV}")

# 8. Species Annotation Status Tracker
print("\n" + "=" * 50)
print("📊 SPECIES ANNOTATION METRICS")
print("=" * 50)

# Check existing JSON annotations
json_files = glob.glob(os.path.join(IMAGES_OUT_DIR, "*.json"))
annotated_ids = set()
for f in json_files:
    try:
        parts = os.path.basename(f).replace(".json", "").split("_")
        if len(parts) > 1:
            annotated_ids.add(int(parts[1]))
    except ValueError:
        continue

df_extracted["is_annotated"] = df_extracted["individual_id"].isin(annotated_ids)

total_species = df_extracted["species"].nunique()
annotated_species_list = df_extracted[df_extracted["is_annotated"]][
    "species"
].unique()
annotated_species_count = len(annotated_species_list)
unannotated_species_count = total_species - annotated_species_count

pct_annotated = (
    (annotated_species_count / total_species * 100) if total_species else 0.0
)
pct_unannotated = (
    (unannotated_species_count / total_species * 100) if total_species else 0.0
)

print(f"Total unique species in dataset : {total_species}")
print(
    f"Species with at least 1 label   : {annotated_species_count} ({pct_annotated:.1f}%)"
)
print(
    f"Species with 0 labels           : {unannotated_species_count} ({pct_unannotated:.1f}%)"
)
print("=" * 50)

df_unannotated = df_extracted[~df_extracted["is_annotated"]]
unannotated_counts = df_unannotated["species"].value_counts()

print(f"\n🔍 TOP SPECIES LACKING LABELS (Work backlog):")
print("-" * 50)
print(f"{'Species Name':<35} | {'Available Images':<15}")
print("-" * 50)
for sp, count in unannotated_counts.head(20).items():
    print(f"{sp:<35} | {count:<15}")
print("-" * 50)
if len(unannotated_counts) > 20:
    print(f"... and {len(unannotated_counts) - 20} more unannotated species.")