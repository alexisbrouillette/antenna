import os
import re
import json

BOOKS_DIR = "data/reference_books"

TARGET_SPECIES = [
    "Opisthoxia formosante",
    "Choristoneura fumiferana",
    "Campaea perlata",
    "Spilomela discordens",
    "Perola sericea",
    "Gaudeator paidicus",
    "Megalopyge",
    "Megalopyge tharops",
    "Perigramma guatemalaria",
    "Eacles imperialis",
    "Hygrochroa",
    "Perigramma repetita",
    "Ategumia dilecticolor",
    "Phostria dohrnii",
    "Spilomela perspicata",
    "Spilomelinae",
    "Sosxetra grata",
    "Copaxa decrescens",
    "Lepidoptera",
    "Tortricidae",
    "Amaxia"
]

SYNONYMS = {
    "Campaea perlata": ["Metrocampa perlata", "Metrocampa perlaria", "Metrocampa praegrandaria", "perlaria"],
    "Eacles imperialis": ["Basilona imperialis", "Eacles magnifica"],
    "Ategumia dilecticolor": ["Syngamia dilecticolor", "Bocchoris dilecticolor", "A. dilecticolor"],
    "Opisthoxia formosante": ["Phalaena formosante", "Opisthoxia"],
    "Choristoneura fumiferana": ["Harmologa fumiferana", "Archips fumiferana", "Tortrix fumiferana", "H. fumiferana"],
    "Phostria dohrnii": ["Plectrona dohrni", "Phostria dohrni", "dohrni"],
    "Perigramma repetita": ["Perigramma repetita", "repetita"],
    "Perigramma guatemalaria": ["Perigramma guatemalaria", "guatemalaria"],
    "Spilomelinae": ["Spilomela", "Pyraustinae", "Pyralidina"]
}

MORPHOLOGY_KEYWORDS = [
    "antenna", "antennae", "pectinat", "filiform", "wingspan", "expanse", 
    "length", "mm", "inch", "fore wing", "hind wing", "palpi", "palpen", "caterpillar", 
    "larva", "dimorphism", "male", "female", "colour", "color", "spruce bud",
    "sprieten", "vlugt", "fascia", "tornus"
]

def load_books():
    books = {}
    for fname in os.listdir(BOOKS_DIR):
        if fname.endswith(".txt"):
            fpath = os.path.join(BOOKS_DIR, fname)
            with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                books[fname] = f.read()
    return books

def search_taxon_in_text(taxon, text):
    matches = []
    
    # 1. Full queries & synonyms
    queries = [taxon]
    if taxon in SYNONYMS:
        queries.extend(SYNONYMS[taxon])
        
    for q in queries:
        pattern = re.compile(rf'\b{re.escape(q)}\b', re.IGNORECASE)
        for m in pattern.finditer(text):
            start = max(0, m.start() - 250)
            end = min(len(text), m.end() + 750)
            snippet = re.sub(r'\s+', ' ', text[start:end].replace('\n', ' ')).strip()
            matches.append({"query": q, "snippet": snippet, "type": "species_match"})
            
    # 2. Specific epithet lookup if binomial
    parts = taxon.split()
    genus = parts[0]
    epithet = parts[1] if len(parts) > 1 else None
    
    if not matches and epithet:
        # Search for unique epithet
        pattern = re.compile(rf'\b{re.escape(epithet)}\b', re.IGNORECASE)
        for m in pattern.finditer(text):
            start = max(0, m.start() - 200)
            end = min(len(text), m.end() + 600)
            snippet = re.sub(r'\s+', ' ', text[start:end].replace('\n', ' ')).strip()
            # If genus initial or genus is nearby
            if genus.lower() in snippet.lower() or f"{genus[0].lower()}." in snippet.lower():
                matches.append({"query": f"{genus[0]}. {epithet}", "snippet": snippet, "type": "species_match"})
                
    # 3. Genus-level fallback
    if not matches and epithet:
        pattern = re.compile(rf'\b{re.escape(genus)}\b', re.IGNORECASE)
        for m in pattern.finditer(text):
            start = max(0, m.start() - 150)
            end = min(len(text), m.end() + 600)
            snippet = re.sub(r'\s+', ' ', text[start:end].replace('\n', ' ')).strip()
            matches.append({"query": genus, "snippet": snippet, "type": "genus_fallback"})
            if len(matches) >= 3:
                break
                
    return matches

def analyze_coverage():
    print("=" * 80)
    print("📖 ENTOMOLOGICAL IDENTIFICATION BOOKS: COVERAGE ANALYSIS")
    print("=" * 80)
    
    books = load_books()
    print(f"Loaded {len(books)} reference monographs ({sum(len(t) for t in books.values()) / (1024*1024):.2f} MB of text).\n")
    
    results = {}
    
    for taxon in TARGET_SPECIES:
        taxon_matches = []
        for bname, btext in books.items():
            matches = search_taxon_in_text(taxon, btext)
            for m in matches:
                morph_found = [k for k in MORPHOLOGY_KEYWORDS if k in m["snippet"].lower()]
                taxon_matches.append({
                    "book": bname,
                    "query": m["query"],
                    "match_type": m["type"],
                    "morphology_signals": morph_found,
                    "snippet": m["snippet"]
                })
                
        species_hits = [m for m in taxon_matches if m["match_type"] == "species_match"]
        genus_hits = [m for m in taxon_matches if m["match_type"] == "genus_fallback"]
        
        if species_hits:
            status = "DIRECT SPECIES HIT"
            # Prefer match with morphology signals
            with_morph = [m for m in species_hits if m["morphology_signals"]]
            best_match = with_morph[0] if with_morph else species_hits[0]
        elif genus_hits:
            status = "GENUS / HIGHER TAXON HIT"
            with_morph = [m for m in genus_hits if m["morphology_signals"]]
            best_match = with_morph[0] if with_morph else genus_hits[0]
        else:
            status = "NOT FOUND IN DOWNLOADED SET"
            best_match = None
            
        results[taxon] = {
            "status": status,
            "total_matches": len(taxon_matches),
            "best_match": best_match
        }
        
    print(f"{'TAXON':<28} | {'COVERAGE STATUS':<24} | {'BOOK SOURCE':<35} | {'MORPHOLOGY SIGNALS'}")
    print("-" * 115)
    
    direct_count = 0
    genus_count = 0
    missing_count = 0
    
    for taxon, res in results.items():
        st = res["status"]
        bm = res["best_match"]
        if st == "DIRECT SPECIES HIT":
            direct_count += 1
            bname = bm["book"]
            signals = ", ".join(bm["morphology_signals"][:3]) if bm["morphology_signals"] else "none"
            print(f"{taxon:<28} | 🟢 {st:<22} | {bname:<35} | {signals}")
        elif st == "GENUS / HIGHER TAXON HIT":
            genus_count += 1
            bname = bm["book"]
            signals = ", ".join(bm["morphology_signals"][:3]) if bm["morphology_signals"] else "none"
            print(f"{taxon:<28} | 🟡 {st:<22} | {bname:<35} | {signals}")
        else:
            missing_count += 1
            print(f"{taxon:<28} | 🔴 {st:<22} | {'-':<35} | -")
            
    print("=" * 115)
    print(f"📊 SUMMARY:")
    print(f"   • Total target taxa: {len(TARGET_SPECIES)}")
    print(f"   • Direct species matches: {direct_count} ({direct_count / len(TARGET_SPECIES) * 100:.1f}%)")
    print(f"   • Genus / higher rank matches: {genus_count} ({genus_count / len(TARGET_SPECIES) * 100:.1f}%)")
    print(f"   • Total Covered: {direct_count + genus_count} / {len(TARGET_SPECIES)} ({(direct_count + genus_count) / len(TARGET_SPECIES) * 100:.1f}%)")
    print(f"   • Missing: {missing_count}")
    print("=" * 115)
    
    with open("data/book_coverage_report.json", "w", encoding="utf-8") as out_f:
        json.dump(results, out_f, indent=2)

if __name__ == "__main__":
    analyze_coverage()

