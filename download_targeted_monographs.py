import os
import urllib.request
from tqdm import tqdm

ADDITIONAL_BOOKS = {
    "warren_1905_novitates_zoologicae_vol12.txt": {
        "url": "https://archive.org/download/novitateszoologi12lond/novitateszoologi12lond_djvu.txt",
        "title": "Novitates Zoologicae Vol. 12 (Warren, 1905)",
        "taxa": "Perigramma repetita (original species description)"
    },
    "schaus_1927_proc_ent_soc_wash.txt": {
        "url": "https://ia800705.us.archive.org/21/items/biostor-99239/biostor-99239_djvu.txt",
        "title": "Proceedings of the Entomological Society of Washington Vol. 29 (Schaus, 1927)",
        "taxa": "Perigramma guatemalaria (original species description)"
    },
    "snellen_1895_tijdschrift_voor_entomologie_vol38.txt": {
        "url": "https://archive.org/download/tijdschriftvoore381895nede/tijdschriftvoore381895nede_djvu.txt",
        "title": "Tijdschrift voor Entomologie Vol. 38 (Snellen, 1895)",
        "taxa": "Phostria dohrnii / Plectrona dohrni (original species description)"
    }
}

OUTPUT_DIR = "data/reference_books"

def download_additional():
    print("=" * 75)
    print("📚 DOWNLOADING ORIGINAL MONOGRAPHS FOR REMAINING SPECIES")
    print("=" * 75)
    for fname, info in ADDITIONAL_BOOKS.items():
        dest = os.path.join(OUTPUT_DIR, fname)
        if os.path.exists(dest) and os.path.getsize(dest) > 10000:
            print(f"✔ Already downloaded: {info['title']}")
            continue
        print(f"\n⬇️ Downloading: {info['title']}...")
        print(f"   Target: {info['taxa']}")
        req = urllib.request.Request(
            info["url"],
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) LepidopteraLibrary/1.0"}
        )
        with urllib.request.urlopen(req, timeout=60) as resp, open(dest, "wb") as f:
            total = int(resp.info().get("Content-Length", 0))
            with tqdm(total=total, unit="B", unit_scale=True, desc=fname) as pbar:
                while True:
                    chunk = resp.read(64 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
                    pbar.update(len(chunk))
        print(f"   Saved to {dest} ({os.path.getsize(dest) / 1024:.1f} KB)")

if __name__ == "__main__":
    download_additional()

