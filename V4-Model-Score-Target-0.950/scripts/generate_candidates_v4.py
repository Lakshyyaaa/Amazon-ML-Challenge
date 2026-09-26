#!/usr/bin/env python3
"""
generate_candidates_v4.py (V4 10-Route Multi-Modal Blocker)
------------------------------------------------------------
1. Implements 10 complementary blocking routes:
   - Route 1: Acoustic Phonetics (Double Metaphone + Soundex via jellyfish)
   - Route 2: Prefix 3-grams and 4-grams
   - Route 3: Character 4-grams (cg4_) & Word Bi-Grams (bg_)
   - Route 4: Acronyms (acr_)
   - Route 5: Postal/PIN code (pin_) & Building Numbers (num_)
   - Route 6: Locality keywords (aword_) & Number-Word hashes (nw_)
   - Route 7: Pure Spatial Address & Street Composite (spa_{num}_{street})
   - Route 8: Native Indic Script Transliteration (Malayalam, Tamil, Gujarati, etc.)
   - Route 9: Domain & URL Token Extraction (dom_{token})
   - Route 10: "Formerly Known As" / Predecessor Disentangler (fka, dba, aka)
2. Uses dynamic sub-linear IDF damping: weight = 1.0 / (1.0 + N_bin / 100.0)
3. Indexes test_source2.tsv and test_source3.tsv (~9.97M targets).
4. Streams test_source1.tsv (1.73M queries) and generates Top 50 candidates.
5. Saves candidate pool to output/candidate_pairs.tsv.
"""

import os
import re
import gc
import sys
import time
from collections import defaultdict
import numpy as np
import pandas as pd
from unidecode import unidecode
import jellyfish
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate

LEGAL_STOP = {
    'inc', 'incorporated', 'llc', 'ltd', 'limited', 'pvt', 'private', 'corp', 'corporation',
    'co', 'company', 'enterprises', 'enterprise', 'services', 'service', 'solutions',
    'the', 'and', 'of', '&', 'group', 'industries', 'industry', 'holdings', 'holding',
    'sarl', 'sasu', 'sas', 'sci', 'eurl', 'sa', 'snc', 'fils', 'groupe', 'societe', 'france',
    'etablissements', 'ste', 'ets', 'praaivett', 'limittedd', 'elelpi', 'praiveett'
}

STREET_NOISE = {
    'road', 'street', 'avenue', 'drive', 'lane', 'court', 'way', 'rue', 'boulevard',
    'allee', 'st', 'rd', 'ave', 'dr', 'blvd', 'township', 'block', 'plot', 'near',
    'floor', 'flat', 'phase', 'sector', 'nagar', 'colony', 'urban', 'rural', 'post', 'box'
}

def detect_and_transliterate(text):
    for ch in text:
        cp = ord(ch)
        if 0x0900 <= cp <= 0x097F:
            return transliterate(text, sanscript.DEVANAGARI, sanscript.ITRANS)
        elif 0x0A80 <= cp <= 0x0AFF:
            return transliterate(text, sanscript.GUJARATI, sanscript.ITRANS)
        elif 0x0B80 <= cp <= 0x0BFF:
            return transliterate(text, sanscript.TAMIL, sanscript.ITRANS)
        elif 0x0C00 <= cp <= 0x0C7F:
            return transliterate(text, sanscript.TELUGU, sanscript.ITRANS)
        elif 0x0C80 <= cp <= 0x0CFF:
            return transliterate(text, sanscript.KANNADA, sanscript.ITRANS)
        elif 0x0D00 <= cp <= 0x0D7F:
            return transliterate(text, sanscript.MALAYALAM, sanscript.ITRANS)
        elif 0x0980 <= cp <= 0x09FF:
            return transliterate(text, sanscript.BENGALI, sanscript.ITRANS)
    return text

def extract_v4_keys(name, addr, country):
    keys = []
    c = str(country).strip()
    
    # 1. Alias & FKA Disentangler
    name_variants = []
    if name and pd.notna(name):
        raw_n = str(name).strip()
        parts = re.split(r'\b(?:formerly known as|fka|aka|dba|t/a|trading as|d\.b\.a\.|doing business as)\b', raw_n, flags=re.IGNORECASE)
        name_variants = [p.strip() for p in parts if len(p.strip()) >= 2]
        if not name_variants:
            name_variants = [raw_n]
    
    for var in name_variants:
        trans_n = detect_and_transliterate(var)
        n_clean = unidecode(trans_n.lstrip('@#')).lower()
        
        # Domain token extraction
        domain_match = re.search(r'\b([a-zA-Z0-9]+)\.(?:com|org|net|in|fr|co\.in)\b', n_clean)
        if domain_match:
            dom_root = domain_match.group(1)
            keys.append('dom_' + dom_root[:6])
            
        toks = [t for t in re.findall(r'[a-zA-Z0-9]+', n_clean) if len(t) >= 2]
        sig_toks = [t for t in toks if t not in LEGAL_STOP and len(t) >= 3]
        
        for t in sig_toks[:4]:
            keys.append('tok_' + t)
            keys.append('pre3_' + t[:3])
            keys.append('pre4_' + t[:4])
            if t.isalpha():
                keys.append('snd_' + jellyfish.soundex(t))
                keys.append('met_' + jellyfish.metaphone(t))
                
        if len(sig_toks) >= 2:
            keys.append(f'bg_{sig_toks[0][:4]}_{sig_toks[1][:4]}')
            keys.append('acr_' + ''.join([t[0] for t in sig_toks[:4]]))
            
        comp = ''.join(toks)
        if len(comp) >= 5:
            keys.append('comp_' + comp[:6])
            for i in range(0, min(len(comp)-3, 10), 2):
                keys.append('cg4_' + comp[i:i+4])
                
    # 2. Address & Pure Spatial Keys
    if addr and pd.notna(addr):
        a_clean = unidecode(str(addr)).lower()
        pins = re.findall(r'\b\d{5,6}\b', a_clean)
        for p in pins[:2]:
            keys.append('pin_' + p)
            
        nums = re.findall(r'\b0*(\d+)[a-zA-Z]?\b', a_clean)
        for n in nums[:3]:
            keys.append('num_' + n)
            
        a_words = [t for t in re.findall(r'[a-zA-Z]{4,}', a_clean) if t not in STREET_NOISE]
        for w in a_words[:5]:
            keys.append('aword_' + w)
            if nums:
                keys.append(f'nw_{nums[0]}_{w}')
                
        # Pure Spatial Address Route (recovers Trade Aliases & Rebrandings)
        if nums and a_words:
            n0 = nums[0]
            for sw in a_words[:3]:
                keys.append(f'spa_{n0}_{sw}')
                
        if pins and nums:
            keys.append(f'spapin_{pins[0]}_{nums[0]}')
            
    return [(c, k) for k in keys]

def main():
    print("=" * 75)
    print("V4 10-ROUTE MULTI-MODAL CANDIDATE GENERATOR (TARGET RECALL: >= 98.5%)")
    print("=" * 75)
    t0 = time.time()

    test_dir = 'dataset/test'
    out_file = 'output/candidate_pairs.tsv'
    max_cands = 50

    # 1. Build Inverted Index
    index = defaultdict(list)
    total_targets = 0
    t_idx = time.time()

    for path, name in [
        (os.path.join(test_dir, 'test_source2.tsv'), 'Source 2'),
        (os.path.join(test_dir, 'test_source3.tsv'), 'Source 3')
    ]:
        print(f"Reading and indexing {name} ({path})...")
        for chunk in pd.read_csv(path, sep='\t', chunksize=250000):
            for tid, b_name, b_addr, ctry in zip(chunk['entity_id'], chunk['business_name'], chunk['business_address'], chunk['country']):
                total_targets += 1
                for k in extract_v4_keys(b_name, b_addr, ctry):
                    index[k].append(tid)

    print(f"Indexed {total_targets:,} target entities into {len(index):,} unique 10-route bins in {time.time()-t_idx:.2f}s.")

    # 2. Query S1 Entities and Stream Top 50 Candidates
    print(f"\nStreaming Source 1 queries to {out_file} (Top {max_cands} per S1)...")
    s1_path = os.path.join(test_dir, 'test_source1.tsv')
    os.makedirs(os.path.dirname(out_file), exist_ok=True)

    chunk_size = 50000
    total_s1 = 0
    total_cands = 0
    zero_cands = 0
    t_query = time.time()

    with open(out_file, 'w', encoding='utf-8', buffering=4*1024*1024) as out_f:
        out_f.write("source1_entity_id\tcandidate_entity_ids\n")

        for chunk_idx, s1_chunk in enumerate(pd.read_csv(s1_path, sep="\t", chunksize=chunk_size), 1):
            out_lines = []
            for s1_id, s1_name, s1_addr, s1_ctry in zip(
                s1_chunk["entity_id"],
                s1_chunk["business_name"],
                s1_chunk["business_address"],
                s1_chunk["country"]
            ):
                keys = extract_v4_keys(s1_name, s1_addr, s1_ctry)
                cand_scores = defaultdict(float)

                for k in keys:
                    bin_records = index.get(k, [])
                    n_bin = len(bin_records)
                    if n_bin > 0:
                        weight = 1.0 / (1.0 + n_bin / 100.0)
                        for tid in bin_records[:100]:
                            cand_scores[tid] += weight

                sorted_cands = [tid for tid, _ in sorted(cand_scores.items(), key=lambda x: x[1], reverse=True)[:max_cands]]
                n_c = len(sorted_cands)
                total_cands += n_c

                if n_c == 0:
                    zero_cands += 1
                    out_lines.append(f"{s1_id}\t\n")
                else:
                    out_lines.append(f"{s1_id}\t{','.join(sorted_cands)}\n")

            out_f.writelines(out_lines)
            total_s1 += len(s1_chunk)

            if chunk_idx % 4 == 0 or total_s1 >= 1732544:
                elapsed = time.time() - t_query
                pct = total_s1 / 1732544 * 100
                speed = total_s1 / elapsed
                print(f"  Processed {total_s1:,} / 1,732,544 S1 ({pct:.1f}%) | Avg cands/S1: {total_cands/total_s1:.1f} | Speed: {speed:.0f} S1/sec")

    total_time = time.time() - t0
    print("\n" + "=" * 75)
    print(f"V4 CANDIDATE GENERATION COMPLETE IN {total_time:.2f}s ({total_time/60:.1f} minutes):")
    print(f"  Total S1 Entities Processed:    {total_s1:,}")
    print(f"  Total Candidates Retrieved:     {total_cands:,}")
    print(f"  Average Candidates per S1:      {total_cands/total_s1:.2f}")
    print(f"  Zero-Candidate Entities:        {zero_cands:,} ({zero_cands/total_s1*100:.2f}%)")
    # Archive copy
    archive_cand = 'V4-Model-Score-Target-0.950/results/candidate_pairs_v4.tsv'
    os.makedirs(os.path.dirname(archive_cand), exist_ok=True)
    import shutil
    shutil.copyfile(out_file, archive_cand)
    print(f"  Primary Candidate Pool:         {out_file} ({os.path.getsize(out_file)/(1024*1024):.2f} MB)")
    print(f"  Archived V4 Candidate Copy:     {archive_cand} ({os.path.getsize(archive_cand)/(1024*1024):.2f} MB)")
    print("=" * 75)

if __name__ == '__main__':
    main()
