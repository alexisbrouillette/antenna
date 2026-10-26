import glob
import os
import shutil
import sys
import pandas as pd

# ==========================================
# CONFIGURATION
# ==========================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
IMAGES_DIR = os.path.join(DATA_DIR, "antenna_field_images")
CSV_PATH = os.path.join(DATA_DIR, "antenna_processed_dataset.csv")
ACTIVE_SET_DIR = os.path.join(DATA_DIR, "active_annotation_set")

ANNOTATION_TARGET = 10  # Target threshold for completion
BUFFER_PER_SPECIES = 20  # Max candidate images to load per target species


def validate_paths():
    if os.path.abspath(ACTIVE_SET_DIR) == os.path.abspath(IMAGES_DIR):
        raise ValueError(
            "ACTIVE_SET_DIR cannot be the same as IMAGES_DIR to avoid accidental deletion."
        )

    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"❌ Processed dataset CSV not found at: {CSV_PATH}")


def load_dataset_and_summary():
    df = pd.read_csv(CSV_PATH)

    # 1. Identify annotated individual IDs from master directory
    existing_json_files = glob.glob(os.path.join(IMAGES_DIR, "*.json"))
    annotated_ids = set()

    for f in existing_json_files:
        try:
            parts = os.path.basename(f).replace(".json", "").split("_")
            if len(parts) > 1:
                annotated_ids.add(int(parts[1]))
        except ValueError:
            continue

    df["is_annotated"] = df["individual_id"].isin(annotated_ids)

    # 2. Generate baseline species summary
    species_summary = (
        df.groupby("species")
        .agg(
            total=("individual_id", "count"),
            annotated=("is_annotated", "sum"),
        )
        .assign(not_annotated=lambda x: x["total"] - x["annotated"])
        .reset_index()
    )

    # 3. Filter target species meeting criteria
    species_filtered = species_summary[
        (species_summary["annotated"] < ANNOTATION_TARGET)
        & (species_summary["not_annotated"] > BUFFER_PER_SPECIES)
    ]
    target_species = set(species_filtered["species"].values)

    return df, species_summary, target_species


def sync_active_set(df, species_summary, target_species, clean_after=False):
    """
    Checks active set annotations and syncs back species that hit ANNOTATION_TARGET.
    """
    if not os.path.exists(ACTIVE_SET_DIR):
        print(f"ℹ️ Active set directory not found: {ACTIVE_SET_DIR}")
        return

    active_json_files = glob.glob(
        os.path.join(ACTIVE_SET_DIR, "**", "*.json"), recursive=True
    )
    print("=" * 75)
    print(
        f"🔍 Checking {len(active_json_files)} active annotations in workspace..."
    )

    active_annotations_by_sp = {}

    for f in active_json_files:
        base_name = os.path.basename(f)
        try:
            ind_id = int(base_name.replace(".json", "").split("_")[1])
        except (IndexError, ValueError):
            continue

        match = df[df["individual_id"] == ind_id]
        if match.empty:
            print(
                f"⚠ Warning: {base_name} does not match any known individual_id. Archiving to IMAGES_DIR."
            )
            shutil.copy2(f, os.path.join(IMAGES_DIR, base_name))
        else:
            sp = match["species"].iloc[0]
            active_annotations_by_sp.setdefault(sp, []).append((ind_id, f))

    # Evaluate completion status against target threshold
    completed_any = False
    for sp in target_species:
        prev_series = species_summary.loc[
            species_summary["species"] == sp, "annotated"
        ]
        prev_count = prev_series.iloc[0] if not prev_series.empty else 0
        new_annotations = active_annotations_by_sp.get(sp, [])
        total_now = prev_count + len(new_annotations)

        if total_now >= ANNOTATION_TARGET:
            completed_any = True
            print(
                f"🎉 Target Reached: '{sp}' completed with {total_now}/{ANNOTATION_TARGET} annotations."
            )
            print(
                f"   📦 Syncing files for '{sp}' from ACTIVE_SET_DIR to IMAGES_DIR..."
            )

            for ind_id, json_path in new_annotations:
                # Copy JSON
                shutil.copy2(
                    json_path,
                    os.path.join(IMAGES_DIR, os.path.basename(json_path)),
                )

                # Find and copy corresponding image (jpg, png)
                img_dir = os.path.dirname(json_path)
                matching_imgs = glob.glob(
                    os.path.join(img_dir, f"*{ind_id}*.[jJ][pP][gG]")
                ) + glob.glob(os.path.join(img_dir, f"*{ind_id}*.[pP][nN][gG]"))

                for img_path in matching_imgs:
                    shutil.copy2(
                        img_path,
                        os.path.join(IMAGES_DIR, os.path.basename(img_path)),
                    )

    if not completed_any:
        print(
            f"⏳ Status: No target species crossed the >={ANNOTATION_TARGET} threshold yet."
        )
        for sp in target_species:
            prev_series = species_summary.loc[
                species_summary["species"] == sp, "annotated"
            ]
            prev_count = prev_series.iloc[0] if not prev_series.empty else 0
            active_count = len(active_annotations_by_sp.get(sp, []))
            print(
                f"   • {sp}: {prev_count + active_count}/{ANNOTATION_TARGET} annotations completed ({active_count} in current batch)"
            )

    if clean_after:
        shutil.rmtree(ACTIVE_SET_DIR)


def populate_active_set(df, target_species):
    """
    Creates species subdirectories inside ACTIVE_SET_DIR and copies candidate images.
    """
    os.makedirs(ACTIVE_SET_DIR, exist_ok=True)

    print("=" * 75)
    print("🎯 POPULATING FRESH ANNOTATION WORKSPACE")
    print("=" * 75)

    total_copied = 0

    for sp in sorted(target_species):
        safe_sp = sp.replace(" ", "_").replace("/", "_")
        sp_dir = os.path.join(ACTIVE_SET_DIR, safe_sp)
        os.makedirs(sp_dir, exist_ok=True)

        candidates = df[(df["species"] == sp) & (~df["is_annotated"])].head(
            BUFFER_PER_SPECIES
        )

        species_copied = 0
        for _, row in candidates.iterrows():
            ind_id = int(row["individual_id"])
            img_name = f"ind_{ind_id}.jpg"
            src_img_path = os.path.join(IMAGES_DIR, img_name)
            dest_img_path = os.path.join(sp_dir, img_name)

            if os.path.exists(src_img_path):
                shutil.copy2(src_img_path, dest_img_path)
                species_copied += 1

        total_copied += species_copied
        print(
            f"📁 Staged {species_copied} unannotated images for '{sp}' in `{safe_sp}/`"
        )

    print(
        f"\n✅ Ready for annotation: {total_copied} total candidate images staged in {ACTIVE_SET_DIR}"
    )


def create_active_set():
    validate_paths()
    df, species_summary, target_species = load_dataset_and_summary()

    # Step 1: Sync back any completed sets from previous runs and clean up directory
    if os.path.exists(ACTIVE_SET_DIR):
        sync_active_set(
            df, species_summary, target_species, clean_after=True
        )

    # Step 2: Repopulate active directory grouped by species subfolders
    populate_active_set(df, target_species)


def sync_back():
    validate_paths()
    df, species_summary, target_species = load_dataset_and_summary()
    sync_active_set(df, species_summary, target_species, clean_after=False)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "sync":
        sync_back()
    else:
        create_active_set()