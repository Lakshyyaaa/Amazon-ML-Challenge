#!/usr/bin/env python3
"""
run_calibrated_inference_v4.py (Precision-Calibrated V4 Post-Processor)
------------------------------------------------------------------------
Applies Precision-Calibrated Post-Processing to recover Precision on Macro F0.5:
1. Hard Postal Circle Veto (if Indian PIN circles differ, impossible match -> 0.00).
2. Hard Lexical Divergence Veto (ntset < 0.35 and atset < 0.70 -> skip).
3. Calibrated Singleton Confidence Floor (tau_sing = 0.88 to preserve ~5.5% singletons).
4. Strict Multi-Match Acceptance (cutoff = max(0.74, best_score - 0.04)).
5. Balanced Source Caps (max 3 from S2, max 3 from S3).
6. High-throughput vectorized scoring across all 1,732,544 test entities.
"""

import os
import re
import gc
import sys
import time
import json
import numpy as np
import pandas as pd
from unidecode import unidecode
from rapidfuzz import fuzz
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

# Helper functions
sys.path.insert(0, os.path.dirname(__file__))
from generate_candidates_v4 import detect_and_transliterate
from train_pipeline_v4 import compute_24d_features, extract_clean_numbers

def extract_pin_circle(addr):
    if not addr:
        return None
    pins = re.findall(r'\b(\d{2})\d{4}\b', str(addr))
    return pins[0] if pins else None

def main():
    print("=" * 75)
    print("V4 PRECISION-CALIBRATED POST-PROCESSOR & INFERENCE ENGINE")
    print("=" * 75)
    t0 = time.time()

    model_dir = 'V4-Model-Score-Target-0.950/models'
    test_dir = 'dataset/test'
    cand_file = 'output/candidate_pairs.tsv'
    out_file = 'output/matching_results.tsv'
    archive_file = 'V4-Model-Score-Target-0.950/results/matching_results_v4_calibrated.tsv'

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
    print(f"Booster Weights: LGB={w_lgb:.2f}, XGB={w_xgb:.2f}, CAT={w_cat:.2f}")

    # Load test entity tables into memory with PIN circles
    print("\nLoading test entity lookup tables into memory...")
    t_load = time.time()
    s1_lookup = {}
    df_s1 = pd.read_csv(os.path.join(test_dir, 'test_source1.tsv'), sep="\t")
    for eid, name, addr in zip(df_s1["entity_id"], df_s1["business_name"], df_s1["business_address"]):
        n_raw = str(name).strip() if pd.notna(name) else ""
        n_clean = unidecode(n_raw).lower()
        a_clean = unidecode(str(addr)).lower() if pd.notna(addr) else ""
        pin_c = extract_pin_circle(a_clean)
        s1_lookup[eid] = (n_clean, a_clean, pin_c, n_clean)
    del df_s1
    gc.collect()

    target_lookup = {}
    for path, is_s2_flag in [(os.path.join(test_dir, 'test_source2.tsv'), 1.0), (os.path.join(test_dir, 'test_source3.tsv'), 0.0)]:
        df = pd.read_csv(path, sep="\t")
        for eid, name, addr in zip(df["entity_id"], df["business_name"], df["business_address"]):
            n_raw = str(name).strip() if pd.notna(name) else ""
            n_clean = unidecode(n_raw).lower()
            a_clean = unidecode(str(addr)).lower() if pd.notna(addr) else ""
            pin_c = extract_pin_circle(a_clean)
            target_lookup[eid] = (n_clean, a_clean, is_s2_flag, pin_c, n_clean)
        del df
        gc.collect()

    print(f"Entities loaded: S1={len(s1_lookup):,}, Targets={len(target_lookup):,} in {time.time()-t_load:.2f}s")

    # Calibration parameters
    tau_singleton = 0.88   # Preserves ~5.5% true singletons, avoiding zero-score penalties
    tau_match = 0.74       # High-confidence precision floor
    margin_secondary = 0.04 # Tight margin to prevent speculative 5th/6th false positives
    top_cands_eval = 50
    batch_size = 25000

    print(f"\nPrecision-Calibrated Configuration:")
    print(f"  Singleton Confidence Floor: tau >= {tau_singleton:.2f}")
    print(f"  Match Acceptance Floor:     tau >= {tau_match:.2f}")
    print(f"  Secondary Margin Gap:       delta <= {margin_secondary:.2f}")
    print(f"  Hard Spatial Circle Veto:   ENABLED (PIN circle mismatch -> reject)")
    print(f"  Hard Name Divergence Veto:  ENABLED (ntset < 0.35 & atset < 0.70 -> reject)")

    print(f"\nStreaming calibrated evaluation from {cand_file} -> {out_file}...")
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
                    s_name, s_addr, s_pinc, s_trans = s1_lookup.get(sid, ("", "", None, ""))

                    for cid in c_list:
                        if cid in target_lookup:
                            t_name, t_addr, is_s2, t_pinc, t_trans = target_lookup[cid]

                            # 1. Hard Postal Circle Veto
                            if s_pinc and t_pinc and s_pinc != t_pinc:
                                continue

                            # 2. Fast Lexical Pre-Filter
                            ntset = fuzz.token_set_ratio(s_name, t_name) / 100.0
                            if ntset < 0.35:
                                atset = fuzz.token_set_ratio(s_addr, t_addr) / 100.0
                                if atset < 0.70:
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

                        # Calibrated singleton floor
                        if best_score < tau_singleton:
                            singleton_s1 += 1
                            out_lines.append(f"{sid}\t\n")
                            continue

                        # Calibrated secondary multi-match pruning
                        sorted_idx = np.argsort(s_sub)[::-1]
                        s2_selected = []
                        s3_selected = []
                        cutoff = max(tau_match, best_score - margin_secondary)

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
                    print(f"  Processed {total_s1:,} / 1,732,544 S1 ({pct:.1f}%) | Matched: {matched_s1:,} ({matched_s1/total_s1*100:.1f}%) | Singletons: {singleton_s1:,} ({singleton_s1/total_s1*100:.1f}%) | Avg matches/S1: {avg_m:.2f} | Speed: {speed:.0f} S1/sec", flush=True)

        # Remainder batch
        if batch_s1_records:
            pair_feats = []
            s1_slices = []
            cands_meta = []
            for sid, c_list in batch_s1_records:
                start_p = len(pair_feats)
                s_name, s_addr, s_pinc, s_trans = s1_lookup.get(sid, ("", "", None, ""))
                for cid in c_list:
                    if cid in target_lookup:
                        t_name, t_addr, is_s2, t_pinc, t_trans = target_lookup[cid]
                        if s_pinc and t_pinc and s_pinc != t_pinc:
                            continue
                        ntset = fuzz.token_set_ratio(s_name, t_name) / 100.0
                        if ntset < 0.35:
                            atset = fuzz.token_set_ratio(s_addr, t_addr) / 100.0
                            if atset < 0.70:
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
                    if best_score < tau_singleton:
                        singleton_s1 += 1
                        out_lines.append(f"{sid}\t\n")
                        continue
                    sorted_idx = np.argsort(s_sub)[::-1]
                    s2_selected = []
                    s3_selected = []
                    cutoff = max(tau_match, best_score - margin_secondary)
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

    elapsed_total = time.time() - t0
    print(f"\nCalibrated Inference Completed in {elapsed_total:.2f}s ({elapsed_total/60:.1f} minutes):")
    print(f"  Total S1 Entities Processed:    {total_s1:,}")
    print(f"  Matched S1 Entities:            {matched_s1:,} ({matched_s1/total_s1*100:.1f}%)")
    print(f"  Singleton S1 Entities:          {singleton_s1:,} ({singleton_s1/total_s1*100:.1f}%)")
    print(f"  Total Matches Predicted:        {total_matches_predicted:,}")
    print(f"  Average Matches Per Matched S1: {total_matches_predicted/matched_s1 if matched_s1>0 else 0:.2f}")

    # Copy to archive
    import shutil
    shutil.copy2(out_file, archive_file)
    print(f"\nSaved primary submission: {out_file} ({os.path.getsize(out_file)/(1024*1024):.2f} MB)")
    print(f"Saved archive copy:       {archive_file} ({os.path.getsize(archive_file)/(1024*1024):.2f} MB)")

    # Run official validator
    print("\n" + "=" * 75)
    print("RUNNING OFFICIAL SUBMISSION VALIDATOR")
    print("=" * 75)
    import subprocess
    cmd = [
        sys.executable, "utils/validate_submission.py",
        "--matching", out_file,
        "--candidate", cand_file,
        "--test-dir", test_dir
    ]
    val_proc = subprocess.run(cmd, capture_output=True, text=True)
    print(val_proc.stdout)
    if val_proc.stderr:
        print("Validator STDERR:", val_proc.stderr)
    print(f"Validator Status: {'PASS (Exit Code 0)' if val_proc.returncode == 0 else 'FAIL (Exit Code ' + str(val_proc.returncode) + ')'}")
    print("=" * 75)

if __name__ == '__main__':
    main()
