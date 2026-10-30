import os
import re
import json

BOOKS_DIR = "data/reference_books"

TARGET_TAXA = [
    {"name": "Opisthoxia formosante", "rank": "species", "family": "Geometridae", "query": ["Opisthoxia formosante", "Phalaena formosante", "Opisthoxia"]},
    {"name": "Choristoneura fumiferana", "rank": "species", "family": "Tortricidae", "query": ["H. fumiferana", "Harmologa fumiferana", "fumiferana", "Choristoneura"]},
    {"name": "Campaea perlata", "rank": "species", "family": "Geometridae", "query": ["Metrocampa praegrandaria", "Metrocampa perlata", "perlaria", "Campaea"]},
    {"name": "Spilomela discordens", "rank": "species", "family": "Crambidae", "query": ["Spilomela discordens", "discordens", "Spilomela"]},
    {"name": "Perola sericea", "rank": "species", "family": "Limacodidae", "query": ["Perola sericea", "sericea", "Perola"]},
    {"name": "Gaudeator paidicus", "rank": "species", "family": "Erebidae", "query": ["Gaudeator paidicus", "GAUDEATOR PAIDICUS", "Gaudeator"]},
    {"name": "Megalopyge", "rank": "genus", "family": "Megalopygidae", "query": ["MEGALOPYGE", "Megalopyge"]},
    {"name": "Megalopyge tharops", "rank": "species", "family": "Megalopygidae", "query": ["tharops", "Megalopyge tharops", "Megalopyge"]},
    {"name": "Perigramma guatemalaria", "rank": "species", "family": "Geometridae", "query": ["Perigramma guatemalaria", "guatemalaria", "Perigramma"]},
    {"name": "Eacles imperialis", "rank": "species", "family": "Saturniidae", "query": ["Eacles imperialis", "Basilona imperialis", "Eacles magnifica", "Eacles"]},
    {"name": "Hygrochroa", "rank": "genus", "family": "Apatelodidae", "query": ["HYGROCHROA", "Hygrochroa"]},
    {"name": "Perigramma repetita", "rank": "species", "family": "Geometridae", "query": ["Perigramma repetita", "repetita", "Perigramma"]},
    {"name": "Ategumia dilecticolor", "rank": "species", "family": "Crambidae", "query": ["Ategumia dilecticolor", "dilecticolor", "Syngamia dilecticolor", "Sameodes flavibaccata"]},
    {"name": "Phostria dohrnii", "rank": "species", "family": "Crambidae", "query": ["Plectrona dohrni", "dohrni", "Phostria dohrnii", "Phostria"]},
    {"name": "Spilomela perspicata", "rank": "species", "family": "Crambidae", "query": ["Spilomela perspicata", "perspicata", "Spilomela"]},
    {"name": "Spilomelinae", "rank": "subfamily", "family": "Crambidae", "query": ["Spilomela", "Pyraustinae", "Pyralidina"]},
    {"name": "Sosxetra grata", "rank": "species", "family": "Erebidae", "query": ["Sosxetra grata", "Sosxetra", "grata"]},
    {"name": "Copaxa decrescens", "rank": "species", "family": "Saturniidae", "query": ["Copaxa decrescens", "decrescens", "Copaxa"]},
    {"name": "Lepidoptera", "rank": "order", "family": "Insecta", "query": ["Lepidoptera"]},
    {"name": "Tortricidae", "rank": "family", "family": "Tortricoidea", "query": ["TORTRICIDAE", "Tortricidae"]},
    {"name": "Amaxia", "rank": "genus", "family": "Erebidae", "query": ["AMAXIA", "Amaxia"]}
]

def load_books():
    books = {}
    for fname in os.listdir(BOOKS_DIR):
        if fname.endswith(".txt"):
            with open(os.path.join(BOOKS_DIR, fname), "r", encoding="utf-8", errors="ignore") as f:
                books[fname] = f.read()
    return books

def clean_ocr(text):
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def extract_database():
    books = load_books()
    db = {}
    
    for item in TARGET_TAXA:
        name = item["name"]
        queries = item["query"]
        
        best_snippet = None
        best_book = None
        plate_ref = None
        
        # Search across all books
        for bname, btext in books.items():
            for q in queries:
                m = re.search(rf'\b{re.escape(q)}\b', btext, re.IGNORECASE)
                if m:
                    start = max(0, m.start() - 250)
                    end = min(len(btext), m.end() + 900)
                    raw_chunk = btext[start:end]
                    
                    # Extract Plate / Tab reference if present
                    plate_m = re.search(r'\((?:Tab\.|Plate)\s*([IVXLCDM\d.]+)\s*(?:fig\.|figg\.|figure)\s*([\d,\s\-.]+)\)', raw_chunk, re.IGNORECASE)
                    if plate_m and not plate_ref:
                        plate_ref = plate_m.group(0).strip("()")
                    
                    if not best_snippet or len(raw_chunk) > len(best_snippet):
                        best_snippet = raw_chunk
                        best_book = bname
                        
        cleaned_text = clean_ocr(best_snippet) if best_snippet else "Generic entomological baseline."
        
        # Check for venation notation
        has_vein_notation = bool(re.search(r'\bvein\s+\d+|\bveins\s+\d+|[MRCu]\d', cleaned_text, re.IGNORECASE))
        has_spur_notation = bool(re.search(r'\bspur|\bspuren', cleaned_text, re.IGNORECASE))
        has_mm_notation = bool(re.search(r'\b\d+\s*(?:mm|inch|inches)\b|\bexpanse\b|\bvlugt\b', cleaned_text, re.IGNORECASE))
        
        notations = []
        if plate_ref:
            notations.append(f"Illustrated Plate Reference: '{plate_ref}' refers to the hand-painted color lithographic plate and figure in the monograph atlas.")
        if has_vein_notation:
            notations.append("Wing Vein Numbers (e.g. 1-12, M1, Cu1): Refers to standard numerical Comstock-Needham wing venation (1=Anal, 2-4=Cubitals, 5-6=Medials, 7-10=Radials).")
        if has_spur_notation:
            notations.append("Spur Formula: Refers to articulated spines on the tibiae of legs (e.g., 'four spurs' = 2 apical + 2 medial on hind tibia).")
        if has_mm_notation:
            notations.append("Expanse / Wingspan: Physical wingspan measured tip-to-tip across forewings (in millimeters or inches).")
            
        # Trait breakdown tags
        db[name] = {
            "taxon": name,
            "rank": item["rank"],
            "family": item["family"],
            "source_monograph": best_book,
            "plate_illustration": plate_ref if plate_ref else "None cited directly in excerpt",
            "notation_guide": notations,
            "verbatim_excerpt": cleaned_text[:1200]
        }
        
    # Write JSON
    out_json = "data/species_morphology_database.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(db, f, indent=2)
    print(f"✔ Saved full morphology database to '{out_json}'")
    
    # Write Markdown Guide
    out_md = "data/SPECIES_MORPHOLOGY_GUIDE.md"
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("# 🦋 Entomological Species Morphology & Trait Guide\n\n")
        f.write("This guide documents the verbatim morphological descriptions, anatomical notation keys, and plate illustration references extracted from historical taxonomic monographs.\n\n")
        f.write("## 📌 Anatomical Notation Glossary\n")
        f.write("* **Vein Numbers (1–12, M, Cu, R):** Standard Comstock-Needham wing venation system. Veins are numbered from trailing edge to leading edge.\n")
        f.write("* **Tibial Spurs:** Articulated sensory/defensive spines on legs (e.g. 'four spurs' = 2 apical + 2 median).\n")
        f.write("* **Expanse / Vlugt:** Wingspan measured tip-to-tip across outstretched forewings.\n")
        f.write("* **Tab. / Plate (e.g. Tab. XV. fig. 13):** References to original full-color hand-painted lithographic plates in the accompanying monograph atlas.\n\n")
        f.write("---\n\n")
        
        for name, data in db.items():
            f.write(f"### {name} (*{data['rank'].capitalize()}*, {data['family']})\n")
            f.write(f"* **Source Monograph:** `{data['source_monograph']}`\n")
            f.write(f"* **Illustrated Plate Reference:** `{data['plate_illustration']}`\n")
            if data["notation_guide"]:
                f.write("* **Notation Notes:**\n")
                for note in data["notation_guide"]:
                    f.write(f"  * {note}\n")
            f.write(f"\n> **Verbatim Description:**\n> {data['verbatim_excerpt']}\n\n")
            f.write("---\n\n")
            
    print(f"✔ Saved markdown guide to '{out_md}'")

if __name__ == "__main__":
    extract_database()

