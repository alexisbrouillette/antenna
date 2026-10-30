import glob
import json
import os
import shutil
import sys
import pandas as pd

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

    # 2. Check parent folder name
    if folder_name and all_species:
        clean_folder = folder_name.replace("_", " ").lower()
        for sp in all_species:
            if sp.lower() == clean_folder:
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
    - AI annotations are verified only if checked=True or status='verified'.
    """
    try:
        with open(json_path, "r", encoding="utf-8") as fp:
            data = json.load(fp)
        flags = data.get("flags", {})
        is_ai = bool(flags.get("ai_generated"))
        if is_ai:
            is_verified = bool(data.get("checked")) or flags.get("status") == "verified"
            return is_verified, True
        return True, False
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
        elif cmd == "approve" and len(sys.argv) > 2:
            approve_annotations(sys.argv[2])
        else:
            print("Usage:")
            print("  python annotation_backlog.py             (Populate / refresh backlog)")
            print("  python annotation_backlog.py sync        (Check status / dry run)")
            print("  python annotation_backlog.py approve <sp> (Mark folder/species as verified)")
    else:
        create_active_set()
