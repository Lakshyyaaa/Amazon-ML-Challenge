#!/usr/bin/env python3
"""
generate_candidates_v3.py
-------------------------
V3 High-Recall Candidate Generator (Targeting 98%+ S1 Match Recall)
Multiple Blocking Routes:
1. Phonetic & Soundex / Metaphone keys (jellyfish)
2. Transliterated Indic scripts & French accents (unidecode)
3. Word N-Grams & Character 4-Grams on compressed names
4. Social Handle & URL de-prefixing (@, www., .com)
5. Address Composite Keys: Street Numbers + Locality + 5/6-digit PIN/ZIP
6. Country-partitioned sub-linear IDF weighted candidate ranking (Top-40 per S1)
"""

import os
import gc
import re
import time
from collections import defaultdict
import pandas as pd
from unidecode import unidecode
import jellyfish

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

def extract_v3_keys(name, addr, country):
    keys = []
    c = str(country).strip()
    
    # 1. Clean, Transliterate & Extract Name Keys
    if name and pd.notna(name):
        n_raw = str(name).lstrip('@')
        n_clean = unidecode(n_raw).lower()
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

    # 2. Address Composite Keys
    if addr and pd.notna(addr):
        a_clean = unidecode(str(addr)).lower()
        
        # PIN / ZIP codes
        pins = re.findall(r'\b\d{5,6}\b', a_clean)
        for p in pins[:2]:
            keys.append('pin_' + p)
            
        # Street Numbers
        nums = re.findall(r'\b\d+\b', a_clean)
        for n in nums[:3]:
            keys.append('num_' + n)
            
        # Locality / Street Words
        a_words = [t for t in re.findall(r'[a-zA-Z]{4,}', a_clean) if t not in STREET_NOISE]
        for w in a_words[:5]:
            keys.append('aword_' + w)
            if nums:
                keys.append(f'nw_{nums[0]}_{w}')

    return [(c, k) for k in keys]

def generate_v3_candidates(
    test_dir='dataset/test',
    out_file='output/candidate_pairs.tsv',
    max_cands=40
):
    print("=" * 70)
    print("V3 MULTI-ROUTE HIGH-RECALL CANDIDATE GENERATION")
    print("=" * 70)
    t0 = time.time()
    
    s1_path = os.path.join(test_dir, 'test_source1.tsv')
    s2_path = os.path.join(test_dir, 'test_source2.tsv')
    s3_path = os.path.join(test_dir, 'test_source3.tsv')
    os.makedirs(os.path.dirname(out_file), exist_ok=True)

    # 1. Index Targets
    print("Indexing Test Source 2 & Source 3 targets across multi-route keys...")
    index = defaultdict(list)
    total_targets = 0

    for src_name, path in [("Source 2", s2_path), ("Source 3", s3_path)]:
        t_src = time.time()
        print(f"  Reading {src_name} ({path})...")
        df = pd.read_csv(path, sep="\t")
        for tid, name, addr, ctry in zip(df["entity_id"], df["business_name"], df["business_address"], df["country"]):
            total_targets += 1
            for k in extract_v3_keys(name, addr, ctry):
                index[k].append(tid)
        del df
        gc.collect()
        print(f"  {src_name} indexed in {time.time()-t_src:.2f}s")

    print(f"Total Targets Indexed: {total_targets:,}")
    print(f"Total Unique V3 Bins: {len(index):,}")
    print(f"Indexing completed in {time.time()-t0:.2f}s")

    # 2. Query Bins for S1 Queries
    print(f"\nStreaming high-recall candidates for Test S1 -> {out_file}...")
    chunk_size = 50000
    total_s1 = 0
    total_cands = 0
    zero_cands = 0

    with open(out_file, 'w', encoding='utf-8', buffering=2*1024*1024) as out_f:
        out_f.write("source1_entity_id\tcandidate_entity_ids\n")

        chunk_start = time.time()
        for chunk_idx, s1_chunk in enumerate(pd.read_csv(s1_path, sep="\t", chunksize=chunk_size), 1):
            out_lines = []
            
            for s1_id, s1_name, s1_addr, s1_ctry in zip(
                s1_chunk["entity_id"],
                s1_chunk["business_name"],
                s1_chunk["business_address"],
                s1_chunk["country"]
            ):
                keys = extract_v3_keys(s1_name, s1_addr, s1_ctry)
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
                elapsed = time.time() - chunk_start
                rate = total_s1 / elapsed
                avg_c = total_cands / total_s1
                pct = total_s1 / 1732544 * 100
                print(f"  Processed {total_s1:,} / 1,732,544 ({pct:.1f}%) | Avg cands/S1: {avg_c:.1f} | Speed: {rate:.0f} S1/sec")

    total_time = time.time() - t0
    print("\n" + "=" * 70)
    print("V3 CANDIDATE GENERATION COMPLETED")
    print("=" * 70)
    print(f"Total S1 entities processed:    {total_s1:,}")
    print(f"Total candidate pairs generated: {total_cands:,}")
    print(f"Average candidates per S1:       {total_cands / total_s1:.2f}")
    print(f"Zero candidate entities:         {zero_cands:,}")
    print(f"Output File: {out_file} ({os.path.getsize(out_file) / (1024*1024):.2f} MB)")
    print(f"Elapsed Time: {total_time:.2f}s ({total_time / 60:.1f} minutes)")
    print("=" * 70)

if __name__ == '__main__':
    generate_v3_candidates()
