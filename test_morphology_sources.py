import json
import re
import urllib.parse
import urllib.request

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) LepidopteraMorphologyResearch/1.0 (contact: alexis@example.com)"

def fetch_json(url, timeout=10):
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode('utf-8'))
    except Exception as e:
        return {"error": str(e)}

# --- 1. GBIF DESCRIPTIONS ---
def query_gbif(taxon_name):
    encoded = urllib.parse.quote(taxon_name)
    match_url = f"https://api.gbif.org/v1/species/match?name={encoded}"
    match_res = fetch_json(match_url)
    
    if "error" in match_res or "usageKey" not in match_res:
        return {"status": "not_found", "data": []}
    
    usage_key = match_res["usageKey"]
    desc_url = f"https://api.gbif.org/v1/species/{usage_key}/descriptions"
    desc_res = fetch_json(desc_url)
    
    descriptions = []
    if "results" in desc_res:
        for r in desc_res["results"]:
            d_type = r.get("type", "general")
            desc = r.get("description", "")
            source = r.get("source", "GBIF Checklist")
            if desc:
                # Clean HTML tags if present
                clean_desc = re.sub(r'<[^>]+>', '', desc).strip()
                descriptions.append({
                    "type": d_type,
                    "source": source,
                    "text": clean_desc
                })
    return {
        "status": "success" if descriptions else "no_descriptions",
        "scientific_name": match_res.get("scientificName"),
        "taxon_key": usage_key,
        "descriptions": descriptions
    }

# --- 2. WIKIPEDIA (Summary & Morphology Sections) ---
def query_wikipedia(taxon_name):
    # Try exact title or clean spaces
    title = taxon_name.replace(" ", "_")
    
    # 1. Page Summary
    summary_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{urllib.parse.quote(title)}"
    summary_res = fetch_json(summary_url)
    
    # 2. Full Extract to get section texts
    extract_url = f"https://en.wikipedia.org/w/api.php?action=query&prop=extracts&explaintext=1&titles={urllib.parse.quote(taxon_name)}&format=json"
    extract_res = fetch_json(extract_url)
    
    full_text = ""
    pages = extract_res.get("query", {}).get("pages", {})
    for pid, pdata in pages.items():
        if pid != "-1":
            full_text = pdata.get("extract", "")
            
    summary_text = summary_res.get("extract", "")
    
    # Extract specific morphology sections if found
    morph_sections = []
    if full_text:
        sections = re.split(r'\n==+\s*(.*?)\s*==+\n', full_text)
        for i in range(1, len(sections), 2):
            sec_title = sections[i].lower()
            sec_body = sections[i+1].strip()
            if any(k in sec_title for k in ["description", "identification", "morphology", "adult", "caterpillar", "larva", "wing", "size"]):
                morph_sections.append({"section": sections[i], "text": sec_body})
                
    return {
        "status": "success" if (summary_text or full_text) else "not_found",
        "summary": summary_text,
        "morphology_sections": morph_sections,
        "full_length": len(full_text)
    }

# --- 3. WIKIDATA (Traits & Claims) ---
def query_wikidata(taxon_name):
    search_url = f"https://www.wikidata.org/w/api.php?action=wbsearchentities&search={urllib.parse.quote(taxon_name)}&language=en&format=json"
    search_res = fetch_json(search_url)
    
    if "error" in search_res or not search_res.get("search"):
        return {"status": "not_found"}
        
    top_entity = search_res["search"][0]
    entity_id = top_entity["id"]
    description = top_entity.get("description", "")
    
    return {
        "status": "success",
        "entity_id": entity_id,
        "label": top_entity.get("label"),
        "description": description
    }

# --- 4. EOL (Encyclopedia of Life) ---
def query_eol(taxon_name):
    search_url = f"https://eol.org/api/search/1.0.json?q={urllib.parse.quote(taxon_name)}&page=1"
    search_res = fetch_json(search_url)
    
    if "error" in search_res or not search_res.get("results"):
        return {"status": "not_found", "descriptions": []}
        
    page_id = search_res["results"][0]["id"]
    details_url = f"https://eol.org/api/pages/1.0/{page_id}.json?text_page=1&details=true"
    details_res = fetch_json(details_url)
    
    texts = []
    data_objects = details_res.get("dataObjects", [])
    for obj in data_objects:
        if obj.get("dataType") == "http://purl.org/dc/dcmitype/Text":
            title = obj.get("title", "")
            desc = obj.get("description", "")
            clean_desc = re.sub(r'<[^>]+>', '', desc).strip()
            texts.append({"title": title, "text": clean_desc})
            
    return {
        "status": "success" if texts else "no_texts",
        "eol_id": page_id,
        "texts": texts
    }

# --- 5. BOLD SYSTEMS (Barcode of Life) ---
def query_bold(taxon_name):
    encoded = urllib.parse.quote(taxon_name)
    tax_url = f"http://v4.boldsystems.org/index.php/API_Tax/TaxonSearch?taxName={encoded}"
    tax_res = fetch_json(tax_url)
    
    # Specimen metadata sample (for morphological notes/sex/stage)
    specimen_url = f"http://v4.boldsystems.org/index.php/API_Public/specimen?taxon={encoded}&format=json"
    spec_res = fetch_json(specimen_url, timeout=12)
    
    spec_records = []
    if isinstance(spec_res, dict) and "bold_records" in spec_res:
        records = spec_res["bold_records"].get("records", {})
        if isinstance(records, dict):
            for rec_id, rec in list(records.items())[:5]: # take sample 5
                spec_records.append({
                    "processid": rec.get("processid"),
                    "sex": rec.get("sex"),
                    "lifestage": rec.get("lifestage"),
                    "notes": rec.get("notes"),
                    "institution": rec.get("institution_storing")
                })
                
    return {
        "status": "success" if (tax_res and "error" not in tax_res) else "not_found",
        "taxonomy": tax_res,
        "sample_specimens": spec_records,
        "total_specimens_checked": len(spec_records)
    }

# --- 6. PLAZI / TREATMENTBANK ---
def query_plazi(taxon_name):
    encoded = urllib.parse.quote(taxon_name)
    url = f"http://treatment.plazi.org/GgServer/search?taxonomicName={encoded}&format=json"
    res = fetch_json(url, timeout=10)
    
    treatments = []
    if isinstance(res, dict) and "results" in res:
        treatments = res.get("results", [])
    elif isinstance(res, list):
        treatments = res
        
    return {
        "status": "success" if treatments else "not_found",
        "treatments_count": len(treatments),
        "treatments": treatments[:3]
    }

def test_species(species_name):
    print("=" * 80)
    print(f"🔬 TESTING MULTI-SOURCE EXTRACTION FOR: '{species_name}'")
    print("=" * 80)
    
    # 1. GBIF
    gbif = query_gbif(species_name)
    print(f"\n[1] GBIF Status: {gbif['status']}")
    if gbif.get("descriptions"):
        for d in gbif["descriptions"][:2]:
            print(f"   ↳ Source: {d['source']} ({d['type']})")
            print(f"     Excerpt: {d['text'][:250]}...\n")
            
    # 2. Wikipedia
    wiki = query_wikipedia(species_name)
    print(f"\n[2] Wikipedia Status: {wiki['status']}")
    if wiki.get("summary"):
        print(f"   ↳ Summary Excerpt: {wiki['summary'][:250]}...")
    if wiki.get("morphology_sections"):
        for sec in wiki["morphology_sections"]:
            print(f"   ↳ Morphology Section [{sec['section']}]: {sec['text'][:250]}...\n")
            
    # 3. Wikidata
    wdata = query_wikidata(species_name)
    print(f"\n[3] Wikidata Status: {wdata['status']} | Description: {wdata.get('description')}")
    
    # 4. EOL
    eol = query_eol(species_name)
    print(f"\n[4] EOL Status: {eol['status']} (Found {len(eol.get('texts', []))} text objects)")
    if eol.get("texts"):
        for t in eol["texts"][:2]:
            print(f"   ↳ {t['title']}: {t['text'][:250]}...\n")
            
    # 5. BOLD Systems
    bold = query_bold(species_name)
    print(f"\n[5] BOLD Systems Status: {bold['status']} | Sample specimens: {len(bold.get('sample_specimens', []))}")
    if bold.get("sample_specimens"):
        print(f"   ↳ Sample specimen data: {bold['sample_specimens'][:2]}")
        
    # 6. Plazi TreatmentBank
    plazi = query_plazi(species_name)
    print(f"\n[6] Plazi Status: {plazi['status']} | Treatments found: {plazi.get('treatments_count')}")

if __name__ == "__main__":
    for sp in ["Choristoneura fumiferana", "Opisthoxia formosante"]:
        test_species(sp)

