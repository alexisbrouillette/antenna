import os
import urllib.request
from tqdm import tqdm

BOOKS = {
    # Central America & Panama (Neotropics) - The Complete Heterocera Series
    "biologia_centrali_americana_vol1.txt": {
        "url": "https://archive.org/download/mobotbca_15_01_00/mobotbca_15_01_00_djvu.txt",
        "title": "Biologia Centrali-Americana: Heterocera Vol. 1 (Druce)",
        "region": "Central America & Panama",
        "taxa": "Saturniidae (Copaxa, Eacles), Megalopygidae, Limacodidae (Perola), Arctiidae"
    },
    "biologia_centrali_americana_vol2.txt": {
        "url": "https://archive.org/download/mobotbca_15_02_00/mobotbca_15_02_00_djvu.txt",
        "title": "Biologia Centrali-Americana: Heterocera Vol. 2 (Druce)",
        "region": "Central America & Panama",
        "taxa": "Geometridae (Opisthoxia, Perigramma), Noctuidae, Erebidae (Gaudeator, Sosxetra)"
    },
    "biologia_centrali_americana_vol4.txt": {
        "url": "https://archive.org/download/mobotbca_15_04_00/mobotbca_15_04_00_djvu.txt",
        "title": "Biologia Centrali-Americana: Heterocera Vol. 4 (Walsingham)",
        "region": "Central America & Panama",
        "taxa": "Pyralidina, Crambidae (Spilomela, Ategumia, Phostria), Tortricidae, Tineina"
    },
    # North America, Canada & Quebec
    "forbes_lepidoptera_new_york_neighboring.txt": {
        "url": "https://archive.org/download/lepidopteraofnew0000will/lepidopteraofnew0000will_djvu.txt",
        "title": "The Lepidoptera of New York and Neighboring States (Forbes, Cornell)",
        "region": "Northeastern North America / Quebec & Ontario",
        "taxa": "Tortricidae (Choristoneura), Pyraloids (Spilomelinae), Bombyces, Primitive forms"
    },
    "forbes_field_tables_lepidoptera.txt": {
        "url": "https://archive.org/download/fieldtableslepi00forbgoog/fieldtableslepi00forbgoog_djvu.txt",
        "title": "Field Tables of Lepidoptera (Forbes)",
        "region": "North America",
        "taxa": "Diagnostic morphological keys and family tables"
    },
    "holland_the_moth_book.txt": {
        "url": "https://archive.org/download/mothbookpopularg00holla/mothbookpopularg00holla_djvu.txt",
        "title": "The Moth Book: North America (W.J. Holland)",
        "region": "North America / Canada",
        "taxa": "Geometridae (Campaea), Saturniidae (Eacles), and all North American families"
    },
    "dyar_lepidoptera_panama_canal_zone.txt": {
        "url": "https://archive.org/download/biostor-58710/biostor-58710_djvu.txt",
        "title": "Lepidoptera of the Smithsonian Biological Survey of the Panama Canal Zone (Dyar, 1914)",
        "region": "Panama & Canal Zone / Barro Colorado",
        "taxa": "Gaudeator paidicus, Ategumia/Syngamia dilecticolor, Spilomela, Crambidae, Arctiinae"
    }
}

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data/reference_books")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def download_file(filename, info):
    dest_path = os.path.join(OUTPUT_DIR, filename)
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 100000:
        print(f"✔ Already downloaded: {info['title']} ({os.path.getsize(dest_path) / 1024:.1f} KB)")
        return
        
    print(f"\n⬇️ Downloading: {info['title']}...")
    print(f"   Region: {info['region']}")
    print(f"   Coverage: {info['taxa']}")
    
    req = urllib.request.Request(
        info["url"],
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) LepidopteraLibrary/1.0"}
    )
    
    with urllib.request.urlopen(req, timeout=60) as response, open(dest_path, "wb") as out_file:
        total_size = int(response.info().get("Content-Length", 0))
        with tqdm(total=total_size, unit="B", unit_scale=True, desc=filename) as pbar:
            while True:
                chunk = response.read(64 * 1024)
                if not chunk:
                    break
                out_file.write(chunk)
                pbar.update(len(chunk))
                
    print(f"   Saved to {dest_path} ({os.path.getsize(dest_path) / 1024:.1f} KB)")

if __name__ == "__main__":
    print("=" * 75)
    print("📚 DOWNLOADING ENTOMOLOGICAL IDENTIFICATION MONOGRAPHS")
    print("=" * 75)
    for fname, meta in BOOKS.items():
        download_file(fname, meta)
    print("\n✨ All reference books downloaded successfully to data/reference_books/!")
