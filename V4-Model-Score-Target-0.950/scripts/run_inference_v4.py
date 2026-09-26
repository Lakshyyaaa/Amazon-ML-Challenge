#!/usr/bin/env python3
"""
run_inference_v4.py (V4 Test Set Inference Engine)
---------------------------------------------------
1. Streams test_source1.tsv and output/candidate_pairs.tsv across all 1,732,544 entities.
2. Extracts 20-D structural features (spatial keys, Indic transliteration, num hierarchy, PIN).
3. Applies fast lexical pre-filter to prune non-viable candidates in microseconds.
4. Predicts with 3-booster regularized ensemble (LightGBM, XGBoost, CatBoost).
5. Applies calibrated confidence floor (tau >= 0.50) and per-source caps (max 3 from S2/S3).
6. Saves primary submission to output/matching_results.tsv and archives to V4 results.
7. Validates submission via utils/validate_submission.py.
"""

import os
import re
import gc
import sys
import time
import json
import shutil
import subprocess
import numpy as np
import pandas as pd
from unidecode import unidecode
from rapidfuzz import fuzz
import jellyfish
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

# Import helpers from training pipeline
sys.path.insert(0, os.path.dirname(__file__))
from generate_candidates_v4 import detect_and_transliterate
from train_pipeline_v4 import compute_24d_features, extract_clean_numbers

def main():
    print("=" * 75)
    print("V4 INFERENCE PIPELINE: 20-D FEATURES & CASCADED ENSEMBLE")
    print("=" * 75)
    t0 = time.time()

    model_dir = 'V4-Model-Score-Target-0.950/models'
    test_dir = 'dataset/test'
    cand_file = 'output/candidate_pairs.tsv'
    out_file = 'output/matching_results.tsv'

    print("Loading V4 3-booster models...")
    lgb_model = lgb.Booster(model_file=os.path.join(model_dir, 'v4_lightgbm.txt'))
    xgb_model = xgb.Booster()
    xgb_model.load_model(os.path.join(model_dir, 'v4_xgboost.json'))
    cat_model = CatBoostClassifier()
    cat_model.load_model(os.path.join(model_dir, 'v4_catboost.cbm'))

    with open(os.path.join(model_dir, 'v4_ensemble_config.json')) as f:
        cfg = json.load(f)
    weights = cfg.get('weights', {'lightgbm': 0.35, 'xgboost': 0.35, 'catboost': 0.30})
    w_lgb, w_xgb, w_cat = weights['lightgbm'], weights['xgboost'], weights['catboost']
    tau_opt = cfg.get('optimal_threshold', 0.77)
    print(f"Loaded Models | Weights: {weights} | Optimal Threshold: {tau_opt:.3f}")

    # Load test entity tables into memory
    print("\nLoading test entity lookup tables into memory...")
    t_load = time.time()
    s1_lookup = {}
    df_s1 = pd.read_csv(os.path.join(test_dir, 'test_source1.tsv'), sep="\t")
    for eid, name, addr in zip(df_s1["entity_id"], df_s1["business_name"], df_s1["business_address"]):
        n_raw = str(name).strip() if pd.notna(name) else ""
        n_trans = unidecode(detect_and_transliterate(n_raw)).lower()
        n_clean = unidecode(n_raw).lower()
        a_clean = unidecode(str(addr)).lower() if pd.notna(addr) else ""
        s1_lookup[eid] = (n_clean, a_clean, n_trans)
    del df_s1
    gc.collect()

    target_lookup = {}
    for path, is_s2_flag in [(os.path.join(test_dir, 'test_source2.tsv'), 1.0), (os.path.join(test_dir, 'test_source3.tsv'), 0.0)]:
        df = pd.read_csv(path, sep="\t")
        for eid, name, addr in zip(df["entity_id"], df["business_name"], df["business_address"]):
            n_raw = str(name).strip() if pd.notna(name) else ""
            n_trans = unidecode(detect_and_transliterate(n_raw)).lower()
            n_clean = unidecode(n_raw).lower()
            a_clean = unidecode(str(addr)).lower() if pd.notna(addr) else ""
            target_lookup[eid] = (n_clean, a_clean, is_s2_flag, n_trans)
        del df
        gc.collect()

    print(f"Entities loaded: S1={len(s1_lookup):,}, Targets={len(target_lookup):,} in {time.time()-t_load:.2f}s")

    # Stream candidates & predict across ALL candidates
    print(f"\nStreaming candidate evaluation from {cand_file} -> {out_file}...")
    top_cands_eval = 50
    batch_size = 20000

    total_s1 = 0
    matched_s1 = 0
    singleton_s1 = 0
    total_matches_predicted = 0
    batch_s1_records = []
    chunk_start = time.time()

    with open(cand_file, 'r', encoding='utf-8') as cand_f, open(out_file, 'w', encoding='utf-8', buffering=4*1024*1024) as out_f:
        next(cand_f) # header
        out_f.write("source1_entity_id\tmatched_entity_ids\n")

        for line in cand_f:
            parts = line.strip().split('\t')
            s1_id = parts[0]
            cands = parts[1].split(',') if (len(parts) > 1 and parts[1]) else []
            batch_s1_records.append((s1_id, cands[:top_cands_eval]))

            if len(batch_s1_records) >= batch_size:
                pair_feats = []
                s1_slices = []
                cands_meta = []

                for sid, c_list in batch_s1_records:
                    start_p = len(pair_feats)
                    s_name, s_addr, s_trans = s1_lookup.get(sid, ("", "", ""))
                    pins_a = re.findall(r'\b\d{5,6}\b', s_addr)
                    nums_a = extract_clean_numbers(s_addr)

                    for cid in c_list:
                        if cid in target_lookup:
                            t_name, t_addr, is_s2, t_trans = target_lookup[cid]
                            ntset = fuzz.token_set_ratio(s_name, t_name) / 100.0
                            atset = fuzz.token_set_ratio(s_addr, t_addr) / 100.0
                            pins_b = re.findall(r'\b\d{5,6}\b', t_addr)
                            pin_m = 1.0 if (pins_a and pins_b and pins_a[0] == pins_b[0]) else 0.0
                            nums_b = extract_clean_numbers(t_addr)
                            num_m = 1.0 if (nums_a and nums_b and (set(nums_a) & set(nums_b))) else 0.0

                            # Early heuristic pre-filter
                            if ntset < 0.16 and atset < 0.16 and pin_m == 0.0 and num_m == 0.0:
                                continue

                            feats = compute_24d_features(s_name, s_addr, t_name, t_addr, s_trans, t_trans, is_s2)
                            pair_feats.append(feats)
                            cands_meta.append((cid, is_s2))

                    end_p = len(pair_feats)
                    s1_slices.append((sid, start_p, end_p))

                if pair_feats:
                    X_mat = np.array(pair_feats, dtype=np.float32)
                    p_lgb = lgb_model.predict(X_mat)
                    p_xgb = xgb_model.predict(xgb.DMatrix(X_mat))
                    p_cat = cat_model.predict_proba(X_mat)[:, 1]
                    scores = (w_lgb * p_lgb) + (w_xgb * p_xgb) + (w_cat * p_cat)
                else:
                    scores = np.array([])

                out_lines = []
                for sid, start_p, end_p in s1_slices:
                    if start_p == end_p:
                        singleton_s1 += 1
                        out_lines.append(f"{sid}\t\n")
                    else:
                        c_sub = cands_meta[start_p:end_p]
                        s_sub = scores[start_p:end_p]
                        best_idx = np.argmax(s_sub)
                        best_score = s_sub[best_idx]

                        # Confidence floor for singleton preservation
                        if best_score < 0.50:
                            singleton_s1 += 1
                            out_lines.append(f"{sid}\t\n")
                            continue

                        sorted_idx = np.argsort(s_sub)[::-1]
                        s2_selected = []
                        s3_selected = []
                        cutoff = max(0.55, best_score - 0.08)

                        for idx in sorted_idx:
                            cid, is_s2 = c_sub[idx]
                            sc_val = s_sub[idx]
                            if sc_val >= cutoff:
                                if is_s2 == 1.0 and len(s2_selected) < 3:
                                    s2_selected.append(cid)
                                elif is_s2 == 0.0 and len(s3_selected) < 3:
                                    s3_selected.append(cid)

                        final_matches = s2_selected + s3_selected
                        if final_matches:
                            matched_s1 += 1
                            total_matches_predicted += len(final_matches)
                            out_lines.append(f"{sid}\t{','.join(final_matches)}\n")
                        else:
                            singleton_s1 += 1
                            out_lines.append(f"{sid}\t\n")

                out_f.writelines(out_lines)
                total_s1 += len(batch_s1_records)
                batch_s1_records = []

                if total_s1 % 200000 == 0 or total_s1 >= 1732544:
                    elapsed = time.time() - chunk_start
                    speed = total_s1 / elapsed
                    pct = total_s1 / 1732544 * 100
                    avg_m = total_matches_predicted / matched_s1 if matched_s1 > 0 else 0
                    print(f"  Processed {total_s1:,} / 1,732,544 S1 ({pct:.1f}%) | Matched: {matched_s1:,} ({matched_s1/total_s1*100:.1f}%) | Singletons: {singleton_s1:,} ({singleton_s1/total_s1*100:.1f}%) | Avg matches/S1: {avg_m:.2f} | Speed: {speed:.0f} S1/sec")

        # Remainder batch
        if batch_s1_records:
            pair_feats = []
            s1_slices = []
            cands_meta = []
            for sid, c_list in batch_s1_records:
                start_p = len(pair_feats)
                s_name, s_addr, s_trans = s1_lookup.get(sid, ("", "", ""))
                pins_a = re.findall(r'\b\d{5,6}\b', s_addr)
                nums_a = extract_clean_numbers(s_addr)
                for cid in c_list:
                    if cid in target_lookup:
                        t_name, t_addr, is_s2, t_trans = target_lookup[cid]
                        ntset = fuzz.token_set_ratio(s_name, t_name) / 100.0
                        atset = fuzz.token_set_ratio(s_addr, t_addr) / 100.0
                        pins_b = re.findall(r'\b\d{5,6}\b', t_addr)
                        pin_m = 1.0 if (pins_a and pins_b and pins_a[0] == pins_b[0]) else 0.0
                        nums_b = extract_clean_numbers(t_addr)
                        num_m = 1.0 if (nums_a and nums_b and (set(nums_a) & set(nums_b))) else 0.0

                        if ntset < 0.16 and atset < 0.16 and pin_m == 0.0 and num_m == 0.0:
                            continue

                        feats = compute_24d_features(s_name, s_addr, t_name, t_addr, s_trans, t_trans, is_s2)
                        pair_feats.append(feats)
                        cands_meta.append((cid, is_s2))

                end_p = len(pair_feats)
                s1_slices.append((sid, start_p, end_p))

            if pair_feats:
                X_mat = np.array(pair_feats, dtype=np.float32)
                p_lgb = lgb_model.predict(X_mat)
                p_xgb = xgb_model.predict(xgb.DMatrix(X_mat))
                p_cat = cat_model.predict_proba(X_mat)[:, 1]
                scores = (w_lgb * p_lgb) + (w_xgb * p_xgb) + (w_cat * p_cat)
            else:
                scores = np.array([])

            out_lines = []
            for sid, start_p, end_p in s1_slices:
                if start_p == end_p:
                    singleton_s1 += 1
                    out_lines.append(f"{sid}\t\n")
                else:
                    c_sub = cands_meta[start_p:end_p]
                    s_sub = scores[start_p:end_p]
                    best_idx = np.argmax(s_sub)
                    best_score = s_sub[best_idx]
                    if best_score < 0.50:
                        singleton_s1 += 1
                        out_lines.append(f"{sid}\t\n")
                        continue
                    sorted_idx = np.argsort(s_sub)[::-1]
                    s2_selected = []
                    s3_selected = []
                    cutoff = max(0.55, best_score - 0.08)
                    for idx in sorted_idx:
                        cid, is_s2 = c_sub[idx]
                        sc_val = s_sub[idx]
                        if sc_val >= cutoff:
                            if is_s2 == 1.0 and len(s2_selected) < 3:
                                s2_selected.append(cid)
                            elif is_s2 == 0.0 and len(s3_selected) < 3:
                                s3_selected.append(cid)
                    final_matches = s2_selected + s3_selected
                    if final_matches:
                        matched_s1 += 1
                        total_matches_predicted += len(final_matches)
                        out_lines.append(f"{sid}\t{','.join(final_matches)}\n")
                    else:
                        singleton_s1 += 1
                        out_lines.append(f"{sid}\t\n")

            out_f.writelines(out_lines)
            total_s1 += len(batch_s1_records)

    total_time = time.time() - t0
    avg_matches = total_matches_predicted / matched_s1 if matched_s1 > 0 else 0
    print(f"\nV4 Inference Completed in {total_time:.2f}s ({total_time/60:.1f} minutes):")
    print(f"  Total S1 Entities Processed:    {total_s1:,}")
    print(f"  Matched S1 Entities:            {matched_s1:,} ({matched_s1/total_s1*100:.1f}%)")
    print(f"  Singleton S1 Entities:          {singleton_s1:,} ({singleton_s1/total_s1*100:.1f}%)")
    print(f"  Total Matches Predicted:        {total_matches_predicted:,}")
    print(f"  Average Matches Per Matched S1: {avg_matches:.2f}")

    # Archive copy
    archive_copy = 'V4-Model-Score-Target-0.950/results/matching_results_v4.tsv'
    os.makedirs(os.path.dirname(archive_copy), exist_ok=True)
    shutil.copyfile(out_file, archive_copy)
    print(f"\nSaved primary submission: {out_file} ({os.path.getsize(out_file)/(1024*1024):.2f} MB)")
    print(f"Saved archive copy:       {archive_copy} ({os.path.getsize(archive_copy)/(1024*1024):.2f} MB)")

    # Official Validation
    print("\n" + "=" * 75)
    print("RUNNING OFFICIAL SUBMISSION VALIDATOR")
    print("=" * 75)
    cmd = [
        "python3", "utils/validate_submission.py",
        "--matching", out_file,
        "--candidate", cand_file,
        "--test-dir", test_dir
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    print(res.stdout)
    if res.stderr:
        print("Stderr:", res.stderr)
    print(f"Validator Status: {'PASS (Exit Code 0)' if res.returncode == 0 else f'FAIL (Exit Code {res.returncode})'}")
    print("=" * 75)

if __name__ == '__main__':
    main()
