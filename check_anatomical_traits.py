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

TRAIT_PATTERNS = {
    "wings": [
        r'\bwing', r'\bwings', r'\bforewing', r'\bhindwing', r'\bprimaries\b', 
        r'\bsecondaries\b', r'\bexpanse\b', r'\bwingspan\b', r'\bcosta\b', r'\bapex\b', r'\btornus\b', r'\bvlugt\b'
    ],
    "head": [
        r'\bhead\b', r'\bfrons\b', r'\bvertex\b', r'\bpalpi\b', r'\bpalpen\b', r'\btongue\b', r'\bproboscis\b', r'\beye\b', r'\beyes\b', r'\bkop\b'
    ],
    "antenna": [
        r'\bantenna', r'\bantennae\b', r'\bpectinat', r'\bfiliform\b', r'\bciliat', r'\bsprieten\b', r'\bsetaceous\b', r'\bsegments?\b'
    ],
    "legs": [
        r'\bleg\b', r'\blegs\b', r'\btibia', r'\btars', r'\bspur', r'\bfemur', r'\bfemora\b', r'\bpooten\b', r'\bsporen\b'
    ],
    "body": [
        r'\bbody\b', r'\bthorax\b', r'\babdomen\b', r'\babdominal\b', r'\blength\b', r'\bcollar\b', r'\bpatagia\b', r'\blijf\b'
    ]
}

def load_books():
    books = {}
    for fname in os.listdir(BOOKS_DIR):
        if fname.endswith(".txt"):
            with open(os.path.join(BOOKS_DIR, fname), "r", encoding="utf-8", errors="ignore") as f:
                books[fname] = f.read()
    return books

def analyze_traits():
    books = load_books()
    print("=" * 85)
    print("🔬 ANATOMICAL TRAIT AUDIT ACROSS ALL 21 TAXA (WINGS, HEAD, ANTENNA, LEGS, BODY)")
    print("=" * 85)
    
    audit_results = {}
    
    for taxon in TARGET_SPECIES:
        # Gather all relevant snippets (species + genus if applicable)
        queries = [taxon]
        if taxon in SYNONYMS:
            queries.extend(SYNONYMS[taxon])
        parts = taxon.split()
        genus = parts[0]
        epithet = parts[1] if len(parts) > 1 else None
        
        combined_text = []
        for bname, btext in books.items():
            for q in queries:
                for m in re.finditer(rf'\b{re.escape(q)}\b', btext, re.IGNORECASE):
                    start = max(0, m.start() - 300)
                    end = min(len(btext), m.end() + 1000)
                    combined_text.append(btext[start:end])
            # Also include genus diagnosis if species
            if epithet:
                for m in re.finditer(rf'\b{re.escape(genus)}\b', btext, re.IGNORECASE):
                    start = max(0, m.start() - 200)
                    end = min(len(btext), m.end() + 800)
                    combined_text.append(btext[start:end])
                    if len(combined_text) >= 15:
                        break
                        
        full_species_corpus = " ".join(combined_text)
        
        # Check traits
        taxon_traits = {}
        for trait, patterns in TRAIT_PATTERNS.items():
            found_evidence = []
            for p in patterns:
                matches = re.findall(p, full_species_corpus, re.IGNORECASE)
                if matches:
                    found_evidence.append(matches[0])
            taxon_traits[trait] = {
                "present": bool(found_evidence),
                "keywords": list(set(found_evidence))
            }
            
        audit_results[taxon] = taxon_traits
        
    # Print formatted matrix
    headers = ["TAXON", "WINGS", "HEAD", "ANTENNA", "LEGS", "BODY", "TRAIT COVERAGE"]
    print(f"{headers[0]:<26} | {headers[1]:<7} | {headers[2]:<7} | {headers[3]:<7} | {headers[4]:<7} | {headers[5]:<7} | {headers[6]}")
    print("-" * 88)
    
    total_cells = len(TARGET_SPECIES) * 5
    covered_cells = 0
    
    for taxon, traits in audit_results.items():
        w = "✔" if traits["wings"]["present"] else "✖"
        h = "✔" if traits["head"]["present"] else "✖"
        a = "✔" if traits["antenna"]["present"] else "✖"
        l = "✔" if traits["legs"]["present"] else "✖"
        b = "✔" if traits["body"]["present"] else "✖"
        
        count = sum(1 for t in ["wings", "head", "antenna", "legs", "body"] if traits[t]["present"])
        covered_cells += count
        pct = (count / 5) * 100
        
        print(f"{taxon:<26} |   {w}   |   {h}   |   {a}   |   {l}   |   {b}   | {count}/5 ({pct:.0f}%)")
        
    print("=" * 88)
    print(f"📊 SUMMARY OF ANATOMICAL TRAIT COVERAGE:")
    print(f"   • Total trait data points: {covered_cells} / {total_cells} ({covered_cells / total_cells * 100:.1f}%)")
    print(f"   • Wings coverage: {sum(1 for t in audit_results.values() if t['wings']['present'])} / 21 ({sum(1 for t in audit_results.values() if t['wings']['present'])/21*100:.1f}%)")
    print(f"   • Head/Palpi coverage: {sum(1 for t in audit_results.values() if t['head']['present'])} / 21 ({sum(1 for t in audit_results.values() if t['head']['present'])/21*100:.1f}%)")
    print(f"   • Antenna coverage: {sum(1 for t in audit_results.values() if t['antenna']['present'])} / 21 ({sum(1 for t in audit_results.values() if t['antenna']['present'])/21*100:.1f}%)")
    print(f"   • Legs/Spurs coverage: {sum(1 for t in audit_results.values() if t['legs']['present'])} / 21 ({sum(1 for t in audit_results.values() if t['legs']['present'])/21*100:.1f}%)")
    print(f"   • Body/Thorax/Abdomen coverage: {sum(1 for t in audit_results.values() if t['body']['present'])} / 21 ({sum(1 for t in audit_results.values() if t['body']['present'])/21*100:.1f}%)")
    print("=" * 88)
    
    with open("data/anatomical_traits_audit.json", "w", encoding="utf-8") as out_f:
        json.dump(audit_results, out_f, indent=2)

if __name__ == "__main__":
    analyze_traits()

