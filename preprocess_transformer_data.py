#!/usr/bin/env python3
"""
Preprocess the complete Amazon ML Challenge dataset for Transformer training
using Name + Address blocking/binning.

Columns created:
- S1_ID
- Target_ID
- Target_Source
- text_a: "Business Name: {S1_Name} Address: {S1_Address} Country: {S1_Country}"
- text_b: "Business Name: {Target_Name} Address: {Target_Address} Country: {Target_Country}"
- label: 1 for known matches, 0 for non-matches
"""

import os
import sys
import gc
import re
import time
import random
from collections import defaultdict
import pandas as pd

# Set fixed seed for reproducibility
random.seed(42)

# Common business suffixes and stop words to remove for name blocking
LEGAL_STOP_WORDS = {
    'inc', 'incorporated', 'llc', 'ltd', 'limited', 'pvt', 'private', 'corp', 'corporation',
    'co', 'company', 'enterprises', 'enterprise', 'services', 'service', 'solutions',
    'the', 'and', 'of', '&', 'group', 'industries', 'industry', 'holdings', 'holding'
}

def extract_name_key(name):
    """Extract primary significant word from business name."""
    if not name or pd.isna(name):
        return ''
    tokens = re.findall(r'[a-zA-Z0-9]+', str(name).lower())
    sig_tokens = [t for t in tokens if t not in LEGAL_STOP_WORDS and len(t) >= 3]
    return sig_tokens[0] if sig_tokens else (tokens[0] if tokens else '')

def extract_addr_key(addr, country):
    """Extract postal code or prominent location token from address."""
    if not addr or pd.isna(addr):
        return ''
    addr_str = str(addr)
    if country == 'India':
        pins = re.findall(r'\b[1-9]\d{5}\b', addr_str)
        if pins:
            return 'pin_' + pins[0]
    elif country == 'US':
        zips = re.findall(r'\b\d{5}\b', addr_str)
        if zips:
            return 'zip_' + zips[0]
    tokens = re.findall(r'[a-zA-Z0-9]+', addr_str.lower())
    if tokens and len(tokens[0]) >= 3:
        return 'addr_' + tokens[0]
    return ''

def format_text(name, addr, country):
    n = '' if pd.isna(name) else str(name).strip()
    a = '' if pd.isna(addr) else str(addr).strip()
    c = '' if pd.isna(country) else str(country).strip()
    return f"Business Name: {n} Address: {a} Country: {c}"

def main():
    start_total_time = time.time()
    
    data_dir = "dataset/train"
    out_dir = "output"
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "transformer_training_pairs.tsv")

    s1_path = os.path.join(data_dir, "train_source1.tsv")
    s2_path = os.path.join(data_dir, "train_source2.tsv")
    s3_path = os.path.join(data_dir, "train_source3.tsv")
    gt_path = os.path.join(data_dir, "train_ground_truth.tsv")

    print("=" * 70)
    print("Pre-processing Amazon ML Challenge Dataset for Transformer Training")
    print("=" * 70)

    # ---------------------------------------------------------
    # 1. Load Ground Truth
    # ---------------------------------------------------------
    print(f"[{time.strftime('%X')}] Loading Ground Truth from {gt_path}...")
    t0 = time.time()
    gt = pd.read_csv(gt_path, sep="\t")
    gt_map = {}
    total_gt_matches = 0
    for s1_id, mids in zip(gt["source1_entity_id"], gt["matched_entity_ids"]):
        if pd.notna(mids) and str(mids).strip():
            matched = [m.strip() for m in str(mids).split(",") if m.strip()]
            gt_map[s1_id] = matched
            total_gt_matches += len(matched)
        else:
            gt_map[s1_id] = []
    del gt
    gc.collect()
    print(f"[{time.strftime('%X')}] Loaded {len(gt_map):,} S1 entities with {total_gt_matches:,} true matches in {time.time()-t0:.2f}s")

    # ---------------------------------------------------------
    # 2. Load & Index Target Sources (S2 and S3)
    # ---------------------------------------------------------
    print(f"\n[{time.strftime('%X')}] Loading and indexing Source 2 & Source 3 targets...")
    t0 = time.time()
    target_text = {}
    name_bins = defaultdict(list)
    addr_bins = defaultdict(list)
    country_pool = defaultdict(list)

    for src_name, path in [("Source 2", s2_path), ("Source 3", s3_path)]:
        t_src = time.time()
        print(f"  Reading {src_name} ({path})...")
        df = pd.read_csv(path, sep="\t")
        for tid, name, addr, ctry in zip(df["entity_id"], df["business_name"], df["business_address"], df["country"]):
            target_text[tid] = format_text(name, addr, ctry)
            country_pool[ctry].append(tid)
            
            nk = extract_name_key(name)
            if nk:
                name_bins[(ctry, nk)].append(tid)
            ak = extract_addr_key(addr, ctry)
            if ak:
                addr_bins[(ctry, ak)].append(tid)
        del df
        gc.collect()
        print(f"  {src_name} loaded and indexed in {time.time()-t_src:.2f}s")

    print(f"[{time.strftime('%X')}] Total Target records indexed: {len(target_text):,}")
    print(f"  Unique (Country, Name) bins: {len(name_bins):,}")
    print(f"  Unique (Country, Address) bins: {len(addr_bins):,}")
    print(f"  Indexing completed in {time.time()-t0:.2f}s")

    # ---------------------------------------------------------
    # 3. Process Source 1 in chunks and generate candidate pairs
    # ---------------------------------------------------------
    print(f"\n[{time.strftime('%X')}] Processing Source 1 in chunks and streaming to {out_file}...")
    chunk_size = 50000
    pos_count = 0
    neg_count = 0
    total_s1_processed = 0

    with open(out_file, "w", encoding="utf-8", buffering=1024 * 1024) as out_f:
        # Write header
        out_f.write("S1_ID\tTarget_ID\tTarget_Source\ttext_a\ttext_b\tlabel\n")

        chunk_start = time.time()
        for chunk_idx, s1_chunk in enumerate(pd.read_csv(s1_path, sep="\t", chunksize=chunk_size), 1):
            out_lines = []
            
            for s1_id, s1_name, s1_addr, s1_ctry in zip(
                s1_chunk["entity_id"],
                s1_chunk["business_name"],
                s1_chunk["business_address"],
                s1_chunk["country"]
            ):
                text_a = format_text(s1_name, s1_addr, s1_ctry)
                true_matches = gt_map.get(s1_id, [])
                true_set = set(true_matches)

                # 3a. Emit all known positive matches (label = 1)
                for tid in true_matches:
                    target_src = "Source 2" if tid.startswith("S2") else "Source 3"
                    text_b = target_text.get(tid, "")
                    out_lines.append(f"{s1_id}\t{tid}\t{target_src}\t{text_a}\t{text_b}\t1\n")
                    pos_count += 1

                # 3b. Generate negative candidates via name + address blocking (label = 0)
                # For balanced 1:1 ratio: n_needed = len(true_matches) for entities with matches, 1 for singletons
                n_needed = len(true_matches) if len(true_matches) > 0 else 1

                s1_nk = extract_name_key(s1_name)
                s1_ak = extract_addr_key(s1_addr, s1_ctry)

                # Collect candidate pool from blocking bins
                candidate_pool = []
                if s1_nk:
                    nb = name_bins.get((s1_ctry, s1_nk))
                    if nb:
                        # Sample up to 25 items from name bin to keep selection fast
                        sample_k = min(len(nb), max(25, n_needed * 3))
                        candidate_pool.extend(random.sample(nb, sample_k))
                if s1_ak:
                    ab = addr_bins.get((s1_ctry, s1_ak))
                    if ab:
                        sample_k = min(len(ab), max(25, n_needed * 3))
                        candidate_pool.extend(random.sample(ab, sample_k))

                # Select n_needed distinct negatives not in true_set
                selected_negs = []
                if candidate_pool:
                    random.shuffle(candidate_pool)
                    for cand_id in candidate_pool:
                        if cand_id not in true_set and cand_id not in selected_negs:
                            selected_negs.append(cand_id)
                            if len(selected_negs) == n_needed:
                                break

                # Fallback to country pool if blocking candidates are fewer than needed
                if len(selected_negs) < n_needed:
                    cpool = country_pool.get(s1_ctry, [])
                    if cpool:
                        sample_pool = random.sample(cpool, min(len(cpool), (n_needed - len(selected_negs)) * 5))
                        for cand_id in sample_pool:
                            if cand_id not in true_set and cand_id not in selected_negs:
                                selected_negs.append(cand_id)
                                if len(selected_negs) == n_needed:
                                    break

                # Emit negative candidates
                for tid in selected_negs:
                    target_src = "Source 2" if tid.startswith("S2") else "Source 3"
                    text_b = target_text.get(tid, "")
                    out_lines.append(f"{s1_id}\t{tid}\t{target_src}\t{text_a}\t{text_b}\t0\n")
                    neg_count += 1

            # Write chunk output
            out_f.writelines(out_lines)
            total_s1_processed += len(s1_chunk)

            if chunk_idx % 5 == 0 or total_s1_processed == 2206821:
                elapsed = time.time() - chunk_start
                rate = total_s1_processed / max(1, elapsed)
                print(f"[{time.strftime('%X')}] Processed {total_s1_processed:,} / 2,206,821 S1 entities "
                      f"({total_s1_processed/2206821*100:.1f}%) | "
                      f"Pairs written: {pos_count + neg_count:,} (Pos: {pos_count:,}, Neg: {neg_count:,}) | "
                      f"Speed: {rate:,.0f} entities/s")

    total_pairs = pos_count + neg_count
    file_size_gb = os.path.getsize(out_file) / (1024 ** 3)
    total_elapsed = time.time() - start_total_time

    print("\n" + "=" * 70)
    print("Preprocessing Completed Successfully!")
    print("=" * 70)
    print(f"Output File:           {out_file}")
    print(f"Output File Size:      {file_size_gb:.2f} GB")
    print(f"Total Candidate Pairs: {total_pairs:,}")
    print(f"Positive Count (1):    {pos_count:,}")
    print(f"Negative Count (0):    {neg_count:,}")
    print(f"Final Dataset Shape:   ({total_pairs:,}, 6)")
    print(f"Total Execution Time:  {total_elapsed/60:.2f} minutes")
    print("=" * 70)

if __name__ == "__main__":
    main()
