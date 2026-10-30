import os
import time
import json
import urllib.request
import urllib.parse
from tqdm import tqdm

HEADERS = {'User-Agent': 'MorphoAI-AntennaResearch/1.0 (contact: alexis@example.com)'}

# Target species needing high-quality images and annotations
TARGET_SPECIES = [
    # Top priority with high local backlog or zero annotations
    "Choristoneura fumiferana",
    "Campaea perlata",
    "Eacles imperialis",
    "Opisthoxia formosante",
    "Copaxa decrescens",
    "Gaudeator paidicus",
    "Perola sericea",
    "Megalopyge tharops",
    "Spilomela perspicata",
    "Spilomela discordens",
    "Perigramma guatemalaria",
    "Perigramma repetita",
    "Phostria dohrnii",
    "Ategumia dilecticolor",
    "Sosxetra grata",
    "Amaxia",
    "Hygrochroa"
]

CURATED_USERS = [
    {"id": "1132388", "name": "Marc Belisle"},
    {"id": "1576809", "name": "Julien Piolain"}
]

SAMPLES_PER_SPECIES = 20
BASE_DIR = "data/inat_high_quality_samples"
METADATA_FILE = "data/inat_samples_metadata.json"

os.makedirs(BASE_DIR, exist_ok=True)

def fetch_json(url, timeout=12):
    req = urllib.request.Request(url, headers=HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode('utf-8'))
    except Exception as e:
        print(f"  [API Error] {url}: {e}")
        return None

def find_taxon_id(species_name):
    url = f"https://api.inaturalist.org/v1/taxa?q={urllib.parse.quote(species_name)}&rank=species,genus"
    data = fetch_json(url)
    if data and data.get("results"):
        for res in data["results"]:
            if res.get("name", "").lower() == species_name.lower():
                return res.get("id"), res.get("name")
        return data["results"][0].get("id"), data["results"][0].get("name")
    return None, species_name

def get_observations_for_species(taxon_id, species_name, target_count=20):
    collected_obs = []
    seen_obs_ids = set()
    
    # Phase 1: Try curated users first (Marc Belisle, Julien Piolain)
    for u in CURATED_USERS:
        u_id = u["id"]
        u_name = u["name"]
        url = f"https://api.inaturalist.org/v1/observations?user_id={u_id}&taxon_id={taxon_id}&has[]=photos&quality_grade=research,needs_id&per_page=30"
        data = fetch_json(url)
        if data and data.get("results"):
            for obs in data["results"]:
                obs_id = obs["id"]
                photos = obs.get("photos", [])
                if obs_id not in seen_obs_ids and photos:
                    seen_obs_ids.add(obs_id)
                    collected_obs.append({
                        "observation_id": obs_id,
                        "species": species_name,
                        "observer": obs.get("user", {}).get("login"),
                        "observer_name": u_name,
                        "source_tier": "Curated User",
                        "quality_grade": obs.get("quality_grade"),
                        "license": photos[0].get("license_code"),
                        # iNat photo URL: replace 'square' with 'original' or 'large'
                        "photo_url": photos[0].get("url", "").replace("square", "large"),
                        "observed_on": obs.get("observed_on"),
                        "place_guess": obs.get("place_guess")
                    })
                    if len(collected_obs) >= target_count:
                        return collected_obs
        time.sleep(0.2)

    # Phase 2: If we still need more samples, expand to all Research Grade across iNaturalist
    needed = target_count - len(collected_obs)
    if needed > 0:
        url = f"https://api.inaturalist.org/v1/observations?taxon_id={taxon_id}&has[]=photos&quality_grade=research&per_page=50"
        data = fetch_json(url)
            
        if data and data.get("results"):
            for obs in data["results"]:
                obs_id = obs["id"]
                photos = obs.get("photos", [])
                if obs_id not in seen_obs_ids and photos:
                    seen_obs_ids.add(obs_id)
                    collected_obs.append({
                        "observation_id": obs_id,
                        "species": species_name,
                        "observer": obs.get("user", {}).get("login"),
                        "observer_name": obs.get("user", {}).get("name"),
                        "source_tier": "Global Research Grade",
                        "quality_grade": obs.get("quality_grade"),
                        "license": photos[0].get("license_code"),
                        "photo_url": photos[0].get("url", "").replace("square", "large"),
                        "observed_on": obs.get("observed_on"),
                        "place_guess": obs.get("place_guess")
                    })
                    if len(collected_obs) >= target_count:
                        break
        time.sleep(0.2)
        
    return collected_obs

def download_image(url, dest_path):
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 1000:
        return True
    try:
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=15) as resp, open(dest_path, "wb") as f:
            f.write(resp.read())
        return True
    except Exception as e:
        if os.path.exists(dest_path):
            os.remove(dest_path)
        return False

def run_pipeline():
    print("=" * 80)
    print("📸 INATURALIST HIGH-QUALITY PHOTO HARVESTER (20 SAMPLES PER SPECIES)")
    print("=" * 80)
    
    all_metadata = {}
    
    for sp in TARGET_SPECIES:
        safe_sp = sp.replace(" ", "_")
        sp_dir = os.path.join(BASE_DIR, safe_sp)
        os.makedirs(sp_dir, exist_ok=True)
        
        print(f"\n🔍 Searching iNat for: '{sp}'...")
        taxon_id, matched_name = find_taxon_id(sp)
        if not taxon_id:
            print(f"   ✖ Could not resolve taxon ID on iNaturalist for '{sp}'.")
            continue
            
        print(f"   ↳ Matched Taxon ID: {taxon_id} ('{matched_name}')")
        observations = get_observations_for_species(taxon_id, sp, target_count=SAMPLES_PER_SPECIES)
        print(f"   ↳ Found {len(observations)} candidates (Curated: {sum(1 for o in observations if o['source_tier'] == 'Curated User')}, Research Grade: {sum(1 for o in observations if o['source_tier'] == 'Global Research Grade')})")
        
        # Download images
        successful_downloads = 0
        for idx, obs in enumerate(observations, start=1):
            img_url = obs["photo_url"]
            obs_id = obs["observation_id"]
            img_filename = f"{safe_sp}_{obs_id}_{idx:02d}.jpg"
            dest = os.path.join(sp_dir, img_filename)
            
            if download_image(img_url, dest):
                obs["local_image_path"] = dest
                successful_downloads += 1
                
        all_metadata[sp] = {
            "taxon_id": taxon_id,
            "target": SAMPLES_PER_SPECIES,
            "downloaded": successful_downloads,
            "samples": observations
        }
        print(f"   ✔ Downloaded {successful_downloads}/{len(observations)} high-resolution images to '{sp_dir}/'")

    # Save tracking metadata
    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(all_metadata, f, indent=2)
    print(f"\n✨ Harvester finished! Metadata saved to '{METADATA_FILE}'")

if __name__ == "__main__":
    run_pipeline()

