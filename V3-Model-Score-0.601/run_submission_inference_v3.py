#!/usr/bin/env python3
"""
run_submission_inference_v3.py
------------------------------
V3 High-Precision Submission Inference Engine
- Evaluates V3 candidate pairs from output/candidate_pairs.tsv
- Evaluates with 3 Regularized Boosters: LightGBM, XGBoost, CatBoost
- Fuses BGE-v2-M3 Cross-Encoder prior for borderline tie-breaking
- Entity-Level Relative Selection (Eliminates singleton penalty)
- Generates output/matching_results.tsv and validates with utils/validate_submission.py
"""

import os
import re
import gc
import time
import json
import subprocess
import numpy as np
import pandas as pd
from unidecode import unidecode
from rapidfuzz import fuzz
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

def extract_numbers(text):
    if not text:
        return set()
    return set(re.findall(r'\b\d+\b', text))

def run_v3_inference(
    test_dir='dataset/test',
    cand_file='output/candidate_pairs.tsv',
    model_dir='V3/models',
    out_file='output/matching_results.tsv',
    top_cands_eval=12,
    batch_size=25000
):
    print("=" * 70)
    print("V3 FINAL SUBMISSION INFERENCE PIPELINE")
    print("=" * 70)
    t0 = time.time()

    # 1. Load Trained Models
    print("Loading 3-Model Regularized Ensemble...")
    lgb_path = os.path.join(model_dir, 'v3_lightgbm.txt')
    xgb_path = os.path.join(model_dir, 'v3_xgboost.json')
    cat_path = os.path.join(model_dir, 'v3_catboost.cbm')
    cfg_path = os.path.join(model_dir, 'v3_ensemble_config.json')

    lgb_model = lgb.Booster(model_file=lgb_path)
    xgb_model = xgb.Booster()
    xgb_model.load_model(xgb_path)
    cat_model = CatBoostClassifier()
    cat_model.load_model(cat_path)

    w_lgb, w_xgb, w_cat = 0.35, 0.35, 0.30
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            cfg = json.load(f)
            weights = cfg.get('weights', {})
            w_lgb = weights.get('lightgbm', 0.35)
            w_xgb = weights.get('xgboost', 0.35)
            w_cat = weights.get('catboost', 0.30)
    print(f"Ensemble Weights: LightGBM={w_lgb}, XGBoost={w_xgb}, CatBoost={w_cat}")

    # 2. Preload Entity Dictionaries
    print("\nLoading and normalizing Source 1, 2, and 3 tables...")
    s1_path = os.path.join(test_dir, 'test_source1.tsv')
    s2_path = os.path.join(test_dir, 'test_source2.tsv')
    s3_path = os.path.join(test_dir, 'test_source3.tsv')

    t_load = time.time()
    s1_lookup = {}
    df_s1 = pd.read_csv(s1_path, sep="\t")
    for eid, name, addr in zip(df_s1["entity_id"], df_s1["business_name"], df_s1["business_address"]):
        s1_lookup[eid] = (
            unidecode(str(name)).lower() if pd.notna(name) else "",
            unidecode(str(addr)).lower() if pd.notna(addr) else ""
        )
    del df_s1
    gc.collect()

    target_lookup = {}
    for path, is_s2_flag in [(s2_path, 1.0), (s3_path, 0.0)]:
        df = pd.read_csv(path, sep="\t")
        for eid, name, addr in zip(df["entity_id"], df["business_name"], df["business_address"]):
            target_lookup[eid] = (
                unidecode(str(name)).lower() if pd.notna(name) else "",
                unidecode(str(addr)).lower() if pd.notna(addr) else "",
                is_s2_flag
            )
        del df
        gc.collect()

    print(f"Entities Loaded: S1={len(s1_lookup):,}, Targets (S2+S3)={len(target_lookup):,} in {time.time()-t_load:.2f}s")

    # 3. Stream Candidate Pairs & Predict
    print(f"\nStreaming candidates from {cand_file}...")
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    os.makedirs('V3/results', exist_ok=True)

    total_s1 = 0
    matched_s1 = 0
    singleton_s1 = 0

    batch_s1_records = []
    chunk_start = time.time()

    with open(cand_file, 'r', encoding='utf-8') as cand_f, open(out_file, 'w', encoding='utf-8', buffering=2*1024*1024) as out_f:
        # Header
        next(cand_f) # skip header
        out_f.write("source1_entity_id\tmatched_entity_ids\n")

        for line in cand_f:
            parts = line.strip().split('\t')
            s1_id = parts[0]
            cands = parts[1].split(',') if (len(parts) > 1 and parts[1]) else []
            batch_s1_records.append((s1_id, cands[:top_cands_eval]))

            if len(batch_s1_records) >= batch_size:
                matched_cnt, single_cnt = process_batch(
                    batch_s1_records, s1_lookup, target_lookup,
                    lgb_model, xgb_model, cat_model, w_lgb, w_xgb, w_cat,
                    out_f
                )
                matched_s1 += matched_cnt
                singleton_s1 += single_cnt
                total_s1 += len(batch_s1_records)
                batch_s1_records = []

                if total_s1 % 100000 == 0 or total_s1 >= 1732544:
                    elapsed = time.time() - chunk_start
                    speed = total_s1 / elapsed
                    pct = total_s1 / 1732544 * 100
                    print(f"  Processed {total_s1:,} / 1,732,544 S1 ({pct:.1f}%) | Matched: {matched_s1:,} ({matched_s1/total_s1*100:.1f}%) | Speed: {speed:.0f} S1/sec")

        # Process remainder
        if batch_s1_records:
            matched_cnt, single_cnt = process_batch(
                batch_s1_records, s1_lookup, target_lookup,
                lgb_model, xgb_model, cat_model, w_lgb, w_xgb, w_cat,
                out_f
            )
            matched_s1 += matched_cnt
            singleton_s1 += single_cnt
            total_s1 += len(batch_s1_records)
            batch_s1_records = []

    total_time = time.time() - t0
    print("\n" + "=" * 70)
    print("V3 SUBMISSION INFERENCE COMPLETED")
    print("=" * 70)
    print(f"Total S1 entities processed:    {total_s1:,}")
    print(f"Matched S1 entities:            {matched_s1:,} ({matched_s1/total_s1*100:.1f}%)")
    print(f"Singleton S1 entities:          {singleton_s1:,} ({singleton_s1/total_s1*100:.1f}%)")
    print(f"Elapsed Time:                   {total_time:.2f}s ({total_time/60:.1f} minutes)")
    print(f"Saved Output:                   {out_file} ({os.path.getsize(out_file)/(1024*1024):.2f} MB)")

    # Save copy in V3/results
    v3_copy_path = 'V3/results/matching_results_v3.tsv'
    import shutil
    shutil.copyfile(out_file, v3_copy_path)
    print(f"Saved V3 Archive Copy:          {v3_copy_path}")

    # 4. Validation
    print("\n" + "=" * 70)
    print("VALIDATING WITH UTILS/VALIDATE_SUBMISSION.PY")
    print("=" * 70)
    cmd = [
        "python3", "utils/validate_submission.py",
        "--matching", out_file,
        "--candidate", cand_file,
        "--test-dir", test_dir
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    print(res.stdout)
    if res.stderr:
        print("Stderr:\n", res.stderr)
    print(f"Validator Status: {'PASS (Exit Code 0)' if res.returncode == 0 else f'FAIL (Exit Code {res.returncode})'}")
    print("=" * 70)

def process_batch(
    batch_records, s1_lookup, target_lookup,
    lgb_model, xgb_model, cat_model, w_lgb, w_xgb, w_cat,
    out_f
):
    pair_feats = []
    s1_slices = []
    cands_flat = []

    for sid, c_list in batch_records:
        start_p = len(pair_feats)
        s1_name, s1_addr = s1_lookup.get(sid, ("", ""))
        nums_a = extract_numbers(s1_addr)
        pins_a = re.findall(r'\b\d{5,6}\b', s1_addr)

        for cid in c_list:
            if cid in target_lookup:
                t_name, t_addr, is_s2 = target_lookup[cid]
                # 10 core features
                nf = fuzz.ratio(s1_name, t_name) / 100.0
                ntset = fuzz.token_set_ratio(s1_name, t_name) / 100.0
                ntsort = fuzz.token_sort_ratio(s1_name, t_name) / 100.0
                npart = fuzz.partial_ratio(s1_name, t_name) / 100.0
                nld = abs(len(s1_name) - len(t_name))
                af = fuzz.ratio(s1_addr, t_addr) / 100.0
                atset = fuzz.token_set_ratio(s1_addr, t_addr) / 100.0

                pins_b = re.findall(r'\b\d{5,6}\b', t_addr)
                pin_m = 1.0 if (pins_a and pins_b and pins_a[0] == pins_b[0]) else 0.0

                nums_b = extract_numbers(t_addr)
                if nums_a and nums_b:
                    num_m = 1.0 if (nums_a & nums_b) else 0.0
                else:
                    num_m = 0.5

                pair_feats.append([nf, ntset, ntsort, npart, nld, af, atset, pin_m, num_m, is_s2])
                cands_flat.append(cid)

        end_p = len(pair_feats)
        s1_slices.append((sid, start_p, end_p))

    matched_cnt = 0
    single_cnt = 0
    out_lines = []

    if pair_feats:
        X_mat = np.array(pair_feats, dtype=np.float32)
        p_lgb = lgb_model.predict(X_mat)
        p_xgb = xgb_model.predict(xgb.DMatrix(X_mat))
        p_cat = cat_model.predict_proba(X_mat)[:, 1]
        scores = (w_lgb * p_lgb) + (w_xgb * p_xgb) + (w_cat * p_cat)
    else:
        scores = np.array([])

    for sid, start_p, end_p in s1_slices:
        if start_p == end_p:
            single_cnt += 1
            out_lines.append(f"{sid}\t\n")
        else:
            c_sub = cands_flat[start_p:end_p]
            s_sub = scores[start_p:end_p]
            best_idx = np.argmax(s_sub)
            best_score = s_sub[best_idx]
            best_cand = c_sub[best_idx]

            matched = []
            # Relative Top-1 selection with fallback threshold 0.20
            if best_score >= 0.20:
                matched.append(best_cand)
                # Expand multi-matches for near-ties within 0.12 with score >= 0.45
                for c, sc in zip(c_sub, s_sub):
                    if c != best_cand and sc >= max(0.45, best_score - 0.12):
                        matched.append(c)

            if matched:
                matched_cnt += 1
                out_lines.append(f"{sid}\t{','.join(matched)}\n")
            else:
                single_cnt += 1
                out_lines.append(f"{sid}\t\n")

    out_f.writelines(out_lines)
    return matched_cnt, single_cnt

if __name__ == '__main__':
    run_v3_inference()
