import glob
import json
import os
import shutil
import sys
from collections import Counter
import numpy as np
import pandas as pd
from PIL import Image

# ==========================================
# CONFIGURATION
# ==========================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
FIELD_IMAGES_DIR = os.path.join(DATA_DIR, "antenna_field_images")
INAT_IMAGES_DIR = os.path.join(DATA_DIR, "inat_high_quality_samples")
CSV_PATH = os.path.join(DATA_DIR, "antenna_processed_dataset.csv")
ACTIVE_SET_DIR = os.path.join(DATA_DIR, "active_annotation_set")

ACTIVE_FIELD_DIR = os.path.join(ACTIVE_SET_DIR, "field")
ACTIVE_INAT_DIR = os.path.join(ACTIVE_SET_DIR, "inat")

# Rule: 10 annotations minimum OF EACH type (Field and iNat)
TARGET_PER_TYPE = 10
BUFFER_PER_SPECIES = 20  # Candidates to stage per species folder

EXCLUDE_CLASSES = {
    "Not Lepidoptera", "Not Identifiable", "Unknown", "Not_Lepidoptera",
    "Not Arthropoda", "Moth"
}


def validate_paths():
    if os.path.abspath(ACTIVE_SET_DIR) in [os.path.abspath(FIELD_IMAGES_DIR), os.path.abspath(INAT_IMAGES_DIR)]:
        raise ValueError(
            "ACTIVE_SET_DIR cannot be the same as master data directories to avoid accidental deletion."
        )
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"❌ Processed dataset CSV not found at: {CSV_PATH}")


def resolve_species(filename, folder_name=None, df=None, all_species=None):
    """
    Identifies species and source type for a given file.
    Robust to nested triage subfolders (e.g. clean_to_approve/ and hand_annotate/).
    """
    base = os.path.basename(filename)

    # 1. Field camera trap: ind_<id>
    if base.startswith("ind_"):
        try:
            ind_id = int(base.replace(".json", "").replace(".jpg", "").replace(".png", "").split("_")[1])
            if df is not None:
                m = df[df["individual_id"] == ind_id]
                if not m.empty:
                    return m["species"].iloc[0], ind_id, "field"
            return None, ind_id, "field"
        except (IndexError, ValueError):
            pass

    # 2. Check path components (robust to clean_to_approve and hand_annotate subfolders)
    if all_species:
        parts = os.path.normpath(filename).replace("\\", "/").split("/")
        for p in parts:
            clean_p = p.replace("_", " ").lower()
            for sp in all_species:
                if sp.lower() == clean_p:
                    return sp, None, "inat"

    # 3. Check filename prefix
    if all_species:
        clean_base = base.replace("_", " ").lower()
        for sp in sorted(all_species, key=len, reverse=True):
            if clean_base.startswith(sp.lower()):
                return sp, None, "inat"

    return None, None, "unknown"


def load_dataset_and_summary():
    df = pd.read_csv(CSV_PATH)
    all_dataset_species = set(df["species"].dropna().unique())

    inat_species = set()
    if os.path.exists(INAT_IMAGES_DIR):
        for entry in os.listdir(INAT_IMAGES_DIR):
            if os.path.isdir(os.path.join(INAT_IMAGES_DIR, entry)):
                inat_species.add(entry.replace("_", " "))

    all_known_species = (all_dataset_species.union(inat_species)) - EXCLUDE_CLASSES

    # 1. Count master FIELD annotations (ind_*.json in FIELD_IMAGES_DIR)
    field_jsons = glob.glob(os.path.join(FIELD_IMAGES_DIR, "ind_*.json"))
    field_annotated_ids = set()
    for f in field_jsons:
        try:
            parts = os.path.basename(f).replace(".json", "").split("_")
            if len(parts) > 1:
                field_annotated_ids.add(int(parts[1]))
        except ValueError:
            continue

    df["is_annotated"] = df["individual_id"].isin(field_annotated_ids)

    # 2. Count master iNat annotations (deduplicate by basename)
    inat_jsons = [
        f for f in glob.glob(os.path.join(FIELD_IMAGES_DIR, "*.json"))
        if not os.path.basename(f).startswith("ind_")
    ]
    if os.path.exists(INAT_IMAGES_DIR):
        inat_jsons.extend(glob.glob(os.path.join(INAT_IMAGES_DIR, "**", "*.json"), recursive=True))

    unique_inat_master = {}
    for f in inat_jsons:
        b = os.path.basename(f)
        if b not in unique_inat_master:
            unique_inat_master[b] = f

    inat_annotated_by_sp = {}
    for b, f in unique_inat_master.items():
        sp, _, _ = resolve_species(
            f, folder_name=os.path.basename(os.path.dirname(f)), all_species=all_known_species
        )
        if sp:
            inat_annotated_by_sp[sp] = inat_annotated_by_sp.get(sp, 0) + 1

    # 3. Aggregate separate counts per species
    field_grouped = (
        df.groupby("species")
        .agg(
            field_total=("individual_id", "count"),
            field_annotated=("is_annotated", "sum")
        )
        .reset_index()
    )

    records = []
    for sp in sorted(all_known_species):
        f_row = field_grouped[field_grouped["species"] == sp]
        f_total = int(f_row["field_total"].iloc[0]) if not f_row.empty else 0
        f_ann = int(f_row["field_annotated"].iloc[0]) if not f_row.empty else 0

        safe_sp = sp.replace(" ", "_")
        sp_inat_dir = os.path.join(INAT_IMAGES_DIR, safe_sp)
        inat_total = 0
        if os.path.exists(sp_inat_dir):
            inat_total = len(glob.glob(os.path.join(sp_inat_dir, "*.jpg"))) + len(glob.glob(os.path.join(sp_inat_dir, "*.png")))
        inat_ann = inat_annotated_by_sp.get(sp, 0)

        # A species qualifies as target if either field or inat has < TARGET_PER_TYPE (10)
        # and has candidate images available
        records.append({
            "species": sp,
            "field_total": f_total,
            "field_annotated": f_ann,
            "field_needed": max(0, TARGET_PER_TYPE - f_ann),
            "inat_total": inat_total,
            "inat_annotated": inat_ann,
            "inat_needed": max(0, TARGET_PER_TYPE - inat_ann)
        })

    summary_df = pd.DataFrame(records)

    # Focus on species that have enough data (>20 total observations across dataset or inat)
    target_species = set(
        summary_df[
            ((summary_df["field_needed"] > 0) | (summary_df["inat_needed"] > 0))
            & ((summary_df["field_total"] + summary_df["inat_total"]) > 20)
        ]["species"].values
    )

    return df, summary_df, target_species, all_known_species


def get_annotation_info(json_path):
    """
    Returns (is_verified, is_ai) for an active annotation.
    - Manual annotations are always verified.
    - AI annotations are verified if:
      * checked=True
      * status in ['verified', True]
      * human edited or modified shapes (description doesn't start with '🤖 AI' or is cleared)
    """
    try:
        with open(json_path, "r", encoding="utf-8") as fp:
            data = json.load(fp)
        if data.get("checked") is True:
            return True, bool(data.get("flags", {}).get("ai_generated"))
        flags = data.get("flags", {})
        if not flags or not flags.get("ai_generated"):
            return True, False
        if flags.get("status") in ["verified", True]:
            return True, True
        shapes = data.get("shapes", [])
        if shapes and any(not s.get("description", "").startswith("🤖 AI") for s in shapes):
            return True, True
        return False, True
    except Exception:
        return True, False


def sync_active_set(df, summary_df, target_species, all_known_species, clean_after=False):
    """
    Checks active set annotations in both 'field/' and 'inat/' subdirectories.
    Syncs verified annotations to master repositories immediately (no minimum threshold).
    Distingushes verified/manual annotations from pending AI pre-annotations.
    """
    if not os.path.exists(ACTIVE_SET_DIR):
        print(f"ℹ️ Active set directory not found: {ACTIVE_SET_DIR}")
        return

    print("=" * 85)
    print("📋 SYNCING VERIFIED ANNOTATIONS (SEPARATE FIELD & INAT TRACKS)")
    print("=" * 85)

    # 1. Master state
    master_field_jsons = glob.glob(os.path.join(FIELD_IMAGES_DIR, "ind_*.json"))
    master_field_ids = set()
    for f in master_field_jsons:
        try:
            parts = os.path.basename(f).replace(".json", "").split("_")
            if len(parts) > 1:
                master_field_ids.add(int(parts[1]))
        except ValueError:
            pass

    master_inat_jsons = [
        f for f in glob.glob(os.path.join(FIELD_IMAGES_DIR, "*.json"))
        if not os.path.basename(f).startswith("ind_")
    ]
    if os.path.exists(INAT_IMAGES_DIR):
        master_inat_jsons.extend(glob.glob(os.path.join(INAT_IMAGES_DIR, "**", "*.json"), recursive=True))
    unique_master_inat_basenames = {os.path.basename(f) for f in master_inat_jsons}

    # 2. Scan & sync field active annotations
    active_field_jsons = glob.glob(os.path.join(ACTIVE_FIELD_DIR, "**", "*.json"), recursive=True)
    active_field_by_sp = {}
    pending_ai_field_by_sp = {}
    new_field_synced = 0
    new_field_synced_by_sp = {}

    for f in active_field_jsons:
        folder = os.path.basename(os.path.dirname(f))
        sp, ind_id, _ = resolve_species(f, folder_name=folder, df=df, all_species=all_known_species)
        if sp:
            is_ver, is_ai = get_annotation_info(f)
            if is_ver:
                active_field_by_sp.setdefault(sp, []).append((f, ind_id))
                # Sync immediately to master repository if not already present
                if ind_id not in master_field_ids:
                    base_json = os.path.basename(f)
                    shutil.copy2(f, os.path.join(FIELD_IMAGES_DIR, base_json))
                    img_path = os.path.splitext(f)[0] + ".jpg"
                    if os.path.exists(img_path):
                        shutil.copy2(img_path, os.path.join(FIELD_IMAGES_DIR, os.path.basename(img_path)))
                    master_field_ids.add(ind_id)
                    new_field_synced += 1
                    new_field_synced_by_sp[sp] = new_field_synced_by_sp.get(sp, 0) + 1
            else:
                pending_ai_field_by_sp.setdefault(sp, []).append((f, ind_id))

    # 3. Scan & sync inat active annotations
    active_inat_jsons = glob.glob(os.path.join(ACTIVE_INAT_DIR, "**", "*.json"), recursive=True)
    active_inat_by_sp = {}
    pending_ai_inat_by_sp = {}
    new_inat_synced = 0
    new_inat_synced_by_sp = {}

    for f in active_inat_jsons:
        folder = os.path.basename(os.path.dirname(f))
        sp, _, _ = resolve_species(f, folder_name=folder, df=df, all_species=all_known_species)
        if sp:
            is_ver, is_ai = get_annotation_info(f)
            if is_ver:
                active_inat_by_sp.setdefault(sp, []).append(f)
                base_json = os.path.basename(f)
                # Sync immediately to master repository if not already present
                if base_json not in unique_master_inat_basenames:
                    safe_sp = sp.replace(" ", "_")
                    dest_inat_sp = os.path.join(INAT_IMAGES_DIR, safe_sp)
                    os.makedirs(dest_inat_sp, exist_ok=True)
                    shutil.copy2(f, os.path.join(FIELD_IMAGES_DIR, base_json))
                    shutil.copy2(f, os.path.join(dest_inat_sp, base_json))
                    img_path = os.path.splitext(f)[0] + ".jpg"
                    if os.path.exists(img_path):
                        shutil.copy2(img_path, os.path.join(FIELD_IMAGES_DIR, os.path.basename(img_path)))
                        shutil.copy2(img_path, os.path.join(dest_inat_sp, os.path.basename(img_path)))
                    unique_master_inat_basenames.add(base_json)
                    new_inat_synced += 1
                    new_inat_synced_by_sp[sp] = new_inat_synced_by_sp.get(sp, 0) + 1
            else:
                pending_ai_inat_by_sp.setdefault(sp, []).append(f)

    # 4. Sync reporting
    total_new = new_field_synced + new_inat_synced
    if total_new > 0:
        print(f"📦 SYNCED {total_new} NEW VERIFIED ANNOTATION(S) TO MASTER STORAGE:")
        if new_field_synced > 0:
            print(f"   • Field images synced: {new_field_synced} files -> {FIELD_IMAGES_DIR}")
        if new_inat_synced > 0:
            print(f"   • iNat images synced:  {new_inat_synced} files -> {INAT_IMAGES_DIR} & {FIELD_IMAGES_DIR}")
        synced_species = sorted(set(list(new_field_synced_by_sp.keys()) + list(new_inat_synced_by_sp.keys())))
        for sp in synced_species:
            nf = new_field_synced_by_sp.get(sp, 0)
            ni = new_inat_synced_by_sp.get(sp, 0)
            items = []
            if nf > 0: items.append(f"+{nf} field")
            if ni > 0: items.append(f"+{ni} inat")
            print(f"     ↳ {sp:<25}: {', '.join(items)}")
        print("-" * 85)
    else:
        print("✔ All verified annotations are currently up-to-date in master storage.\n")

    # 5. Status report & milestone checks (target 10 each)
    print(f"{'SPECIES':<28} | {'FIELD ANNOTATIONS':<26} | {'INAT ANNOTATIONS':<26}")
    print("-" * 86)
    for sp in sorted(target_species):
        safe_sp = sp.replace(" ", "_")

        # Total unique verified field annotations
        sp_field_ids = set(df[df["species"] == sp]["individual_id"].dropna().astype(int))
        tot_f = len(sp_field_ids.intersection(master_field_ids))

        # Total unique verified inat annotations
        tot_i = sum(1 for b in unique_master_inat_basenames if b.lower().startswith(safe_sp.lower()))

        pend_f = len(pending_ai_field_by_sp.get(sp, []))
        pend_i = len(pending_ai_inat_by_sp.get(sp, []))

        if tot_f >= TARGET_PER_TYPE:
            f_status = f"✔ {tot_f}/{TARGET_PER_TYPE} (Done)"
        else:
            needed = TARGET_PER_TYPE - tot_f
            extra = f"{needed} needed"
            if pend_f > 0:
                extra += f", {pend_f} AI ready"
            f_status = f"{tot_f}/{TARGET_PER_TYPE} ({extra})"

        if tot_i >= TARGET_PER_TYPE:
            i_status = f"✔ {tot_i}/{TARGET_PER_TYPE} (Done)"
        else:
            needed = TARGET_PER_TYPE - tot_i
            extra = f"{needed} needed"
            if pend_i > 0:
                extra += f", {pend_i} AI ready"
            i_status = f"{tot_i}/{TARGET_PER_TYPE} ({extra})"

        print(f"{sp:<28} | {f_status:<26} | {i_status:<26}")
    print("=" * 86)

    # 6. Safe cleanup if clean_after=True
    if clean_after:
        shutil.rmtree(ACTIVE_SET_DIR)


def populate_active_set(df, summary_df, target_species):
    """
    Creates separate 'field/' and 'inat/' staging folders inside ACTIVE_SET_DIR.
    Stages unannotated candidate images up to BUFFER_PER_SPECIES for each source.
    """
    os.makedirs(ACTIVE_FIELD_DIR, exist_ok=True)
    os.makedirs(ACTIVE_INAT_DIR, exist_ok=True)

    print("\n" + "=" * 85)
    print("🎯 POPULATING SEPARATE ANNOTATION TRACKS (10 MINIMUM FOR EACH)")
    print("=" * 85)

    # Master set of already annotated image stems
    master_field_jsons = glob.glob(os.path.join(FIELD_IMAGES_DIR, "*.json"))
    annotated_stems = {os.path.splitext(os.path.basename(f))[0] for f in master_field_jsons}
    if os.path.exists(INAT_IMAGES_DIR):
        for f in glob.glob(os.path.join(INAT_IMAGES_DIR, "**", "*.json"), recursive=True):
            annotated_stems.add(os.path.splitext(os.path.basename(f))[0])

    total_field_staged = 0
    total_inat_staged = 0

    for sp in sorted(target_species):
        row = summary_df[summary_df["species"] == sp]
        field_needed = int(row["field_needed"].iloc[0]) if not row.empty else 0
        inat_needed = int(row["inat_needed"].iloc[0]) if not row.empty else 0
        safe_sp = sp.replace(" ", "_").replace("/", "_")

        # --- Track 1: Field Images Staging ---
        if field_needed > 0:
            sp_field_dir = os.path.join(ACTIVE_FIELD_DIR, safe_sp)
            os.makedirs(sp_field_dir, exist_ok=True)
            field_candidates = df[(df["species"] == sp) & (~df["is_annotated"])]
            
            staged_f = 0
            for _, r in field_candidates.iterrows():
                ind_id = int(r["individual_id"])
                img_name = f"ind_{ind_id}.jpg"
                stem = f"ind_{ind_id}"
                if stem not in annotated_stems:
                    src = os.path.join(FIELD_IMAGES_DIR, img_name)
                    dst = os.path.join(sp_field_dir, img_name)
                    if os.path.exists(src):
                        shutil.copy2(src, dst)
                        staged_f += 1
                        if staged_f >= BUFFER_PER_SPECIES:
                            break
            total_field_staged += staged_f
        else:
            staged_f = 0

        # --- Track 2: iNaturalist Images Staging ---
        if inat_needed > 0:
            sp_inat_dir = os.path.join(ACTIVE_INAT_DIR, safe_sp)
            os.makedirs(sp_inat_dir, exist_ok=True)
            master_sp_inat = os.path.join(INAT_IMAGES_DIR, safe_sp)
            
            staged_i = 0
            if os.path.exists(master_sp_inat):
                candidates = sorted(
                    glob.glob(os.path.join(master_sp_inat, "*.jpg")) + glob.glob(os.path.join(master_sp_inat, "*.png"))
                )
                for img_path in candidates:
                    stem = os.path.splitext(os.path.basename(img_path))[0]
                    if stem not in annotated_stems:
                        dst = os.path.join(sp_inat_dir, os.path.basename(img_path))
                        shutil.copy2(img_path, dst)
                        staged_i += 1
                        if staged_i >= BUFFER_PER_SPECIES:
                            break
            total_inat_staged += staged_i
        else:
            staged_i = 0

        print(f"📁 '{sp:<26}' staged -> Field: {staged_f:>2} images (`field/{safe_sp}/`) | iNat: {staged_i:>2} images (`inat/{safe_sp}/`)")

    print("\n" + "=" * 85)
    print(f"✅ STAGING COMPLETE:")
    print(f"   • Total Field images staged: {total_field_staged} in {ACTIVE_FIELD_DIR}")
    print(f"   • Total iNat images staged:  {total_inat_staged} in {ACTIVE_INAT_DIR}")
    print(f"   • Grand Total:               {total_field_staged + total_inat_staged} candidate images")
    print("=" * 85)


def replenish_active_set(auto_ai=True):
    """
    1. Syncs all verified annotations to master storage immediately.
    2. Removes only verified/completed images from data/active_annotation_set/ (leaving unreviewed images untouched).
    3. Refills slots up to BUFFER_PER_SPECIES (20) with new unannotated candidate images for species that still need annotations.
    4. Automatically runs AI pre-annotation on newly staged images so they arrive with draft masks.
    """
    validate_paths()
    df, summary_df, target_species, all_known_species = load_dataset_and_summary()

    print("=" * 85)
    print("🔄 REPLENISHING ACTIVE BACKLOG (SYNC -> ARCHIVE VERIFIED -> REFILL CANDIDATES)")
    print("=" * 85)

    # 1. Sync verified to master storage
    sync_active_set(df, summary_df, target_species, all_known_species, clean_after=False)

    # 2. Reload master state after sync
    df, summary_df, target_species, all_known_species = load_dataset_and_summary()
    master_field_jsons = glob.glob(os.path.join(FIELD_IMAGES_DIR, "*.json"))
    annotated_stems = {os.path.splitext(os.path.basename(f))[0] for f in master_field_jsons}
    if os.path.exists(INAT_IMAGES_DIR):
        for f in glob.glob(os.path.join(INAT_IMAGES_DIR, "**", "*.json"), recursive=True):
            annotated_stems.add(os.path.splitext(os.path.basename(f))[0])

    # 3. Evacuate only verified files from active set
    active_jsons = glob.glob(os.path.join(ACTIVE_SET_DIR, "**", "*.json"), recursive=True)
    evacuated = 0
    for jf in active_jsons:
        stem = os.path.splitext(os.path.basename(jf))[0]
        if stem in annotated_stems:
            img_path = os.path.splitext(jf)[0] + ".jpg"
            if os.path.exists(img_path):
                os.remove(img_path)
            os.remove(jf)
            evacuated += 1

    print(f"\n🧹 Evacuated {evacuated} verified image(s) from active set (safely stored in master).")

    # 4. Refill slots up to BUFFER_PER_SPECIES (20) for target species that still need annotations
    total_field_staged = 0
    total_inat_staged = 0

    active_field_stems = {
        os.path.splitext(os.path.basename(f))[0]
        for f in glob.glob(os.path.join(ACTIVE_FIELD_DIR, "**", "*.jpg"), recursive=True)
    }
    active_inat_stems = {
        os.path.splitext(os.path.basename(f))[0]
        for f in glob.glob(os.path.join(ACTIVE_INAT_DIR, "**", "*.jpg"), recursive=True)
    }

    for sp in sorted(target_species):
        row = summary_df[summary_df["species"] == sp]
        field_needed = int(row["field_needed"].iloc[0]) if not row.empty else 0
        inat_needed = int(row["inat_needed"].iloc[0]) if not row.empty else 0
        safe_sp = sp.replace(" ", "_").replace("/", "_")

        # Refill Field track
        if field_needed > 0:
            sp_field_dir = os.path.join(ACTIVE_FIELD_DIR, safe_sp)
            os.makedirs(sp_field_dir, exist_ok=True)
            existing_count = len(glob.glob(os.path.join(sp_field_dir, "*.jpg")))
            slots_to_fill = max(0, BUFFER_PER_SPECIES - existing_count)

            if slots_to_fill > 0:
                field_candidates = df[(df["species"] == sp) & (~df["is_annotated"])]
                staged = 0
                for _, r in field_candidates.iterrows():
                    ind_id = int(r["individual_id"])
                    stem = f"ind_{ind_id}"
                    img_name = f"{stem}.jpg"
                    if stem not in annotated_stems and stem not in active_field_stems:
                        src = os.path.join(FIELD_IMAGES_DIR, img_name)
                        dst = os.path.join(sp_field_dir, img_name)
                        if os.path.exists(src):
                            shutil.copy2(src, dst)
                            active_field_stems.add(stem)
                            staged += 1
                            if staged >= slots_to_fill:
                                break
                total_field_staged += staged

        # Refill iNat track
        if inat_needed > 0:
            sp_inat_dir = os.path.join(ACTIVE_INAT_DIR, safe_sp)
            os.makedirs(sp_inat_dir, exist_ok=True)
            existing_count = len(glob.glob(os.path.join(sp_inat_dir, "*.jpg")))
            slots_to_fill = max(0, BUFFER_PER_SPECIES - existing_count)

            if slots_to_fill > 0:
                master_sp_inat = os.path.join(INAT_IMAGES_DIR, safe_sp)
                staged = 0
                if os.path.exists(master_sp_inat):
                    candidates = sorted(
                        glob.glob(os.path.join(master_sp_inat, "*.jpg")) + glob.glob(os.path.join(master_sp_inat, "*.png"))
                    )
                    for img_path in candidates:
                        stem = os.path.splitext(os.path.basename(img_path))[0]
                        if stem not in annotated_stems and stem not in active_inat_stems:
                            dst = os.path.join(sp_inat_dir, os.path.basename(img_path))
                            shutil.copy2(img_path, dst)
                            active_inat_stems.add(stem)
                            staged += 1
                            if staged >= slots_to_fill:
                                break
                total_inat_staged += staged

    print(f"📦 Replenished candidate images: +{total_field_staged} field, +{total_inat_staged} iNat")

    # 5. Clean up any empty species folders where target is reached
    for base_dir in [ACTIVE_FIELD_DIR, ACTIVE_INAT_DIR]:
        if os.path.exists(base_dir):
            for sp_d in os.listdir(base_dir):
                full_sp_d = os.path.join(base_dir, sp_d)
                if os.path.isdir(full_sp_d) and len(os.listdir(full_sp_d)) == 0:
                    try:
                        os.rmdir(full_sp_d)
                    except OSError:
                        pass

    # 6. Auto run AI pre-annotation on unannotated images
    if auto_ai and (total_field_staged + total_inat_staged) > 0:
        print("\n🤖 Running AI pre-annotation on newly staged images...")
        model_script = os.path.join(SCRIPT_DIR, "cluster_yolo_training", "predict_and_annotate.py")
        weights_path = os.path.join(SCRIPT_DIR, "cluster_yolo_training", "weights", "best.pt")
        if os.path.exists(model_script) and os.path.exists(weights_path):
            import subprocess
            subprocess.run([sys.executable, model_script, "--weights", weights_path, "--target", ACTIVE_SET_DIR])
        else:
            print("ℹ️ Model weights not found; skipping auto-preannotation.")

    print("\n✔ Backlog replenish complete! Open AnyLabeling to review the updated set.")


def score_prediction(result):
    """
    Computes an Annotation Quality Score (0 to 100) based on detection confidence and anatomical sanity.
    Returns: (score, tier, reasons)
    """
    if result.masks is None or len(result.masks) == 0:
        return 0, "hard", ["No parts detected"]
    classes = [result.names[int(c)] for c in result.boxes.cls.cpu().numpy()]
    scores = result.boxes.conf.cpu().numpy()
    mean_conf = float(np.mean(scores))
    counts = Counter(classes)
    penalty = 0
    reasons = []

    # 1. Full-body sanity check
    fb = counts.get("full-body", 0)
    if fb == 1:
        pass
    elif fb == 0:
        penalty += 30
        reasons.append("Missing full-body")
    elif fb > 2:
        penalty += 30
        reasons.append(f"Multiple full-bodies ({fb})")

    # 2. Implausible part counts
    if counts.get("wing", 0) > 4:
        penalty += 25
        reasons.append(f"Too many wings ({counts['wing']})")
    if counts.get("leg", 0) > 6:
        penalty += 20
        reasons.append(f"Too many legs ({counts['leg']})")
    if counts.get("antenna", 0) > 2:
        penalty += 15
        reasons.append(f"Too many antennae ({counts['antenna']})")
    if counts.get("head", 0) > 2:
        penalty += 20
        reasons.append(f"Multiple heads ({counts['head']})")

    # 3. Weak detections (<0.40)
    low_confs = sum(1 for s in scores if s < 0.40)
    if low_confs > 0:
        penalty += min(25, low_confs * 8)
        reasons.append(f"{low_confs} weak parts")

    final_score = max(0, min(100, int(mean_conf * 100 - penalty)))
    tier = "clean" if final_score >= 70 and penalty == 0 else ("review" if final_score >= 45 else "hard")
    return final_score, tier, reasons


def save_triage_annotation(result, img_path, dst_img, dst_json, quality_score, tier, reasons, weights_path):
    shutil.copy2(img_path, dst_img)
    with Image.open(img_path) as im:
        width, height = im.size

    shapes = []
    if result.masks is not None:
        polygons = result.masks.xy
        classes = result.boxes.cls.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()

        for poly, cls_id, score in zip(polygons, classes, confs):
            if len(poly) < 3:
                continue
            label_name = result.names[int(cls_id)]
            points = [[round(float(pt[0]), 2), round(float(pt[1]), 2)] for pt in poly]

            shapes.append({
                "label": label_name,
                "score": round(float(score), 4),
                "points": points,
                "group_id": None,
                "description": f"🤖 AI ({label_name} {score:.0%})",
                "difficult": False,
                "shape_type": "polygon",
                "flags": {"ai_predicted": True},
                "attributes": {},
                "kie_linking": []
            })

    tier_label = "clean_to_approve" if tier == "clean" else "hand_annotate"
    labelme_data = {
        "version": "5.0.1",
        "flags": {
            "ai_generated": True,
            "quality_score": quality_score,
            "quality_tier": tier_label,
            "reasons": reasons,
            "model": os.path.basename(weights_path),
            "status": "needs_human_review"
        },
        "checked": False,
        "shapes": shapes,
        "imagePath": os.path.basename(img_path),
        "imageData": None,
        "imageHeight": height,
        "imageWidth": width
    }

    with open(dst_json, "w", encoding="utf-8") as fp:
        json.dump(labelme_data, fp, indent=2)


def triage_backlog(top_clean=40, bottom_hard=20, max_eval_per_sp=100):
    validate_paths()
    df, summary_df, target_species, all_known_species = load_dataset_and_summary()

    print("=" * 85)
    print(f"🎯 POPULATING TRIAGE BACKLOG (TOP {top_clean} CLEAN & BOTTOM {bottom_hard} HARD PER SPECIES)")
    print("=" * 85)

    # 1. Sync any existing verified annotations first and wipe old active backlog cleanly
    if os.path.exists(ACTIVE_SET_DIR):
        sync_active_set(df, summary_df, target_species, all_known_species, clean_after=True)

    os.makedirs(ACTIVE_FIELD_DIR, exist_ok=True)
    os.makedirs(ACTIVE_INAT_DIR, exist_ok=True)

    # Reload master state
    df, summary_df, target_species, all_known_species = load_dataset_and_summary()
    master_field_jsons = glob.glob(os.path.join(FIELD_IMAGES_DIR, "*.json"))
    annotated_stems = {os.path.splitext(os.path.basename(f))[0] for f in master_field_jsons}
    if os.path.exists(INAT_IMAGES_DIR):
        for f in glob.glob(os.path.join(INAT_IMAGES_DIR, "**", "*.json"), recursive=True):
            annotated_stems.add(os.path.splitext(os.path.basename(f))[0])

    # 2. Load trained model
    weights_path = os.path.join(SCRIPT_DIR, "cluster_yolo_training", "weights", "best.pt")
    assert os.path.exists(weights_path), f"❌ Trained model not found at: {weights_path}"
    print(f"🤖 Loading trained model: {weights_path}\n")
    from ultralytics import YOLO
    model = YOLO(weights_path)

    total_clean_staged = 0
    total_hard_staged = 0
    report_rows = []

    for sp in sorted(target_species):
        row = summary_df[summary_df["species"] == sp]
        field_needed = int(row["field_needed"].iloc[0]) if not row.empty else 0
        inat_needed = int(row["inat_needed"].iloc[0]) if not row.empty else 0
        safe_sp = sp.replace(" ", "_").replace("/", "_")

        # --- Track 1: Field Images Triage ---
        if field_needed > 0:
            sp_clean_dir = os.path.join(ACTIVE_FIELD_DIR, safe_sp, "clean_to_approve")
            sp_hard_dir = os.path.join(ACTIVE_FIELD_DIR, safe_sp, "hand_annotate")
            os.makedirs(sp_clean_dir, exist_ok=True)
            os.makedirs(sp_hard_dir, exist_ok=True)

            field_candidates = df[(df["species"] == sp) & (~df["is_annotated"])]
            cand_files = []
            for _, r in field_candidates.iterrows():
                ind_id = int(r["individual_id"])
                stem = f"ind_{ind_id}"
                if stem not in annotated_stems:
                    fpath = os.path.join(FIELD_IMAGES_DIR, f"{stem}.jpg")
                    if os.path.exists(fpath):
                        cand_files.append(fpath)
                    if len(cand_files) >= max_eval_per_sp:
                        break

            if cand_files:
                eval_results = []
                for fpath in cand_files:
                    res = model.predict(source=fpath, conf=0.25, verbose=False)[0]
                    score, tier, reasons = score_prediction(res)
                    eval_results.append((fpath, score, tier, reasons, res))

                # Sort descending by quality score
                eval_results.sort(key=lambda x: x[1], reverse=True)

                if len(eval_results) >= (top_clean + bottom_hard):
                    clean_set = eval_results[:top_clean]
                    hard_set = eval_results[-bottom_hard:]
                else:
                    n_clean = max(1, int(len(eval_results) * 0.66))
                    clean_set = eval_results[:n_clean]
                    hard_set = eval_results[n_clean:]

                for fpath, sc, tier, reasons, res in clean_set:
                    b = os.path.basename(fpath)
                    stem = os.path.splitext(b)[0]
                    dst_img = os.path.join(sp_clean_dir, b)
                    dst_json = os.path.join(sp_clean_dir, stem + ".json")
                    save_triage_annotation(res, fpath, dst_img, dst_json, sc, "clean", reasons, weights_path)

                for fpath, sc, tier, reasons, res in hard_set:
                    b = os.path.basename(fpath)
                    stem = os.path.splitext(b)[0]
                    dst_img = os.path.join(sp_hard_dir, b)
                    dst_json = os.path.join(sp_hard_dir, stem + ".json")
                    save_triage_annotation(res, fpath, dst_img, dst_json, sc, "hard", reasons, weights_path)

                total_clean_staged += len(clean_set)
                total_hard_staged += len(hard_set)
                avg_clean = np.mean([x[1] for x in clean_set]) if clean_set else 0
                avg_hard = np.mean([x[1] for x in hard_set]) if hard_set else 0
                report_rows.append((f"[FIELD] {sp}", len(cand_files), len(clean_set), avg_clean, len(hard_set), avg_hard))

        # --- Track 2: iNat Images Triage ---
        if inat_needed > 0:
            sp_clean_dir = os.path.join(ACTIVE_INAT_DIR, safe_sp, "clean_to_approve")
            sp_hard_dir = os.path.join(ACTIVE_INAT_DIR, safe_sp, "hand_annotate")
            os.makedirs(sp_clean_dir, exist_ok=True)
            os.makedirs(sp_hard_dir, exist_ok=True)

            master_sp_inat = os.path.join(INAT_IMAGES_DIR, safe_sp)
            cand_files = []
            if os.path.exists(master_sp_inat):
                for fpath in glob.glob(os.path.join(master_sp_inat, "*.jpg")) + glob.glob(os.path.join(master_sp_inat, "*.png")):
                    stem = os.path.splitext(os.path.basename(fpath))[0]
                    if stem not in annotated_stems:
                        cand_files.append(fpath)

            if cand_files:
                eval_results = []
                for fpath in cand_files:
                    res = model.predict(source=fpath, conf=0.25, verbose=False)[0]
                    score, tier, reasons = score_prediction(res)
                    eval_results.append((fpath, score, tier, reasons, res))

                eval_results.sort(key=lambda x: x[1], reverse=True)

                if len(eval_results) >= (top_clean + bottom_hard):
                    clean_set = eval_results[:top_clean]
                    hard_set = eval_results[-bottom_hard:]
                else:
                    n_clean = max(1, int(len(eval_results) * 0.66))
                    clean_set = eval_results[:n_clean]
                    hard_set = eval_results[n_clean:]

                for fpath, sc, tier, reasons, res in clean_set:
                    b = os.path.basename(fpath)
                    stem = os.path.splitext(b)[0]
                    dst_img = os.path.join(sp_clean_dir, b)
                    dst_json = os.path.join(sp_clean_dir, stem + ".json")
                    save_triage_annotation(res, fpath, dst_img, dst_json, sc, "clean", reasons, weights_path)

                for fpath, sc, tier, reasons, res in hard_set:
                    b = os.path.basename(fpath)
                    stem = os.path.splitext(b)[0]
                    dst_img = os.path.join(sp_hard_dir, b)
                    dst_json = os.path.join(sp_hard_dir, stem + ".json")
                    save_triage_annotation(res, fpath, dst_img, dst_json, sc, "hard", reasons, weights_path)

                total_clean_staged += len(clean_set)
                total_hard_staged += len(hard_set)
                avg_clean = np.mean([x[1] for x in clean_set]) if clean_set else 0
                avg_hard = np.mean([x[1] for x in hard_set]) if hard_set else 0
                report_rows.append((f"[INAT ] {sp}", len(cand_files), len(clean_set), avg_clean, len(hard_set), avg_hard))

    print("\n" + "=" * 85)
    print("📊 DATASET-WIDE TRIAGE & PRE-ANNOTATION SUMMARY")
    print("=" * 85)
    print(f"{'SPECIES':<32} | {'EVAL':<5} | {'CLEAN (TOP 40)':<18} | {'HARD (BOTTOM 20)':<18}")
    print("-" * 85)
    for tag, n_eval, n_c, avg_c, n_h, avg_h in report_rows:
        c_str = f"{n_c} (avg {avg_c:.0f}%)"
        h_str = f"{n_h} (avg {avg_h:.0f}%)"
        print(f"{tag:<32} | {n_eval:<5} | {c_str:<18} | {h_str:<18}")
    print("=" * 85)
    print(f"✔ TRIAGE COMPLETE: {total_clean_staged} in 'clean_to_approve/', {total_hard_staged} in 'hand_annotate/'.")
    print("💡 Open AnyLabeling and load the 'clean_to_approve' subfolder for fast 1-click approvals!\n")


def create_active_set():
    validate_paths()
    df, summary_df, target_species, all_known_species = load_dataset_and_summary()

    # Step 1: Sync back completed sets & preserve active progress before clearing
    if os.path.exists(ACTIVE_SET_DIR):
        sync_active_set(
            df, summary_df, target_species, all_known_species, clean_after=True
        )

    # Step 2: Repopulate active directory with separate field and inat folders
    populate_active_set(df, summary_df, target_species)


def approve_annotations(target):
    """
    Marks matching active JSON annotations as checked/verified.
    Can be a species name, a folder path, or a specific image ID / file.
    """
    if not os.path.exists(ACTIVE_SET_DIR):
        print("❌ Active set directory not found.")
        return

    all_jsons = glob.glob(os.path.join(ACTIVE_SET_DIR, "**", "*.json"), recursive=True)
    target_clean = target.replace("\\", "/").lower()

    approved_count = 0
    for jf in all_jsons:
        jf_clean = jf.replace("\\", "/").lower()
        if target_clean in jf_clean:
            try:
                with open(jf, "r", encoding="utf-8") as fp:
                    data = json.load(fp)
                data["checked"] = True
                if "flags" in data and isinstance(data["flags"], dict):
                    data["flags"]["status"] = "verified"
                with open(jf, "w", encoding="utf-8") as fp:
                    json.dump(data, fp, indent=2)
                approved_count += 1
            except Exception as e:
                print(f"⚠️ Error approving {jf}: {e}")

    print(f"✔ Marked {approved_count} annotation(s) as VERIFIED for '{target}'!")
    print("Run './env/bin/python3 annotation_backlog.py sync' to see updated totals.")


def sync_back():
    validate_paths()
    df, summary_df, target_species, all_known_species = load_dataset_and_summary()
    sync_active_set(df, summary_df, target_species, all_known_species, clean_after=False)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        cmd = sys.argv[1].lower()
        if cmd == "sync":
            sync_back()
        elif cmd in ["replenish", "cycle", "refresh"]:
            no_ai = "--no-ai" in sys.argv
            replenish_active_set(auto_ai=not no_ai)
        elif cmd in ["triage", "rank"]:
            triage_backlog(top_clean=40, bottom_hard=20)
        elif cmd == "approve" and len(sys.argv) > 2:
            approve_annotations(sys.argv[2])
        else:
            print("Usage:")
            print("  python annotation_backlog.py triage      (Evaluate all images -> Top 40 in clean/ & Bottom 20 in hard/)")
            print("  python annotation_backlog.py replenish   (Transfer verified -> archive -> refill new images -> pre-annotate)")
            print("  python annotation_backlog.py sync        (Check status / dry run sync to master)")
            print("  python annotation_backlog.py approve <sp> (Mark folder/species as verified)")
            print("  python annotation_backlog.py             (Populate / refresh entire backlog from scratch)")
    else:
        create_active_set()
