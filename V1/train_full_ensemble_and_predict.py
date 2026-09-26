#!/usr/bin/env python3
"""
train_full_ensemble_and_predict.py
-----------------------------------
1. Cleans old matching outputs in output/.
2. Trains LightGBM & XGBoost on 13.1M samples with Apple Silicon acceleration.
3. Evaluates 50-50 ensemble on 2.31M validation samples and finds optimal threshold tau* for Macro F0.5.
4. Saves models locally to models/.
5. Runs streaming batched inference over all 20.7M test candidate pairs in output/candidate_pairs.tsv.
6. Saves predictions to output/matching_results.tsv.
7. Validates submission via utils/validate_submission.py.
"""

import os
import re
import gc
import sys
import time
import json
import subprocess
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, roc_auc_score, f1_score
from rapidfuzz import fuzz
import lightgbm as lgb
import xgboost as xgb
import torch

def compute_f_beta(prec, rec, beta=0.5):
    if prec + rec == 0:
        return 0.0
    b2 = beta ** 2
    return (1 + b2) * (prec * rec) / (b2 * prec + rec)

def main():
    print("=" * 70)
    print("STARTING FULL 15.4M ENSEMBLE TRAINING & INFERENCE PIPELINE")
    print("=" * 70)

    # 1. Clean old matching results
    old_matching = 'output/matching_results.tsv'
    if os.path.exists(old_matching):
        os.remove(old_matching)
        print(f"[CLEANUP] Removed existing {old_matching}")
    for fname in os.listdir('output'):
        if fname.endswith('.csv') and 'matching' in fname.lower():
            p = os.path.join('output', fname)
            os.remove(p)
            print(f"[CLEANUP] Removed {p}")

    mps_avail = torch.backends.mps.is_available()
    cpus = os.cpu_count()
    print(f"[HARDWARE] Apple Silicon CPU Cores: {cpus} | PyTorch MPS: {mps_avail}")

    # 2. Load 15.4M Features & Labels
    print("\n[STEP 1/6] Loading 15.4M pre-extracted feature matrix...")
    t0 = time.time()
    X = np.load('models/features_15m.npy')
    y = np.load('models/labels_15m.npy')
    print(f"Loaded X {X.shape} ({X.nbytes/1e6:.1f} MB) and y {y.shape} ({y.nbytes/1e6:.1f} MB) in {time.time()-t0:.2f}s")
    print(f"Class Balance: {np.mean(y == 1)*100:.2f}% Positives, {np.mean(y == 0)*100:.2f}% Negatives")

    # 3. Train/Val Split (85/15)
    print("\n[STEP 2/6] Splitting Train (85%) / Validation (15%)...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )
    print(f"Train samples: {X_train.shape[0]:,} | Validation samples: {X_val.shape[0]:,}")

    # 4. Train LightGBM
    print("\n[STEP 3/6] Training LightGBM on 13.1M samples...")
    t_lgb = time.time()
    lgb_model = lgb.LGBMClassifier(
        n_estimators=180,
        learning_rate=0.08,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=4
    )
    lgb_model.fit(
        X_train, y_train,
        eval_set=[(X_val[:200000], y_val[:200000])],
        callbacks=[lgb.early_stopping(25, verbose=False), lgb.log_evaluation(period=60)]
    )
    lgb_time = time.time() - t_lgb
    val_preds_lgb = lgb_model.predict_proba(X_val)[:, 1]
    lgb_auc = roc_auc_score(y_val, val_preds_lgb)
    print(f"LightGBM trained in {lgb_time:.2f}s | Val ROC-AUC: {lgb_auc:.4f}")

    # 5. Train XGBoost
    print("\n[STEP 4/6] Training XGBoost on 13.1M samples...")
    t_xgb = time.time()
    xgb_model = xgb.XGBClassifier(
        n_estimators=180,
        learning_rate=0.08,
        max_depth=7,
        tree_method='hist',
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='logloss',
        early_stopping_rounds=25,
        random_state=42,
        n_jobs=4
    )
    xgb_model.fit(
        X_train, y_train,
        eval_set=[(X_val[:200000], y_val[:200000])],
        verbose=60
    )
    xgb_time = time.time() - t_xgb
    val_preds_xgb = xgb_model.predict_proba(X_val)[:, 1]
    xgb_auc = roc_auc_score(y_val, val_preds_xgb)
    print(f"XGBoost trained in {xgb_time:.2f}s | Val ROC-AUC: {xgb_auc:.4f}")

    # 6. Ensemble & Threshold Optimization
    print("\n[STEP 5/6] Evaluating Ensemble on 2.31M Validation Pairs...")
    val_preds_ens = 0.50 * val_preds_lgb + 0.50 * val_preds_xgb
    ens_auc = roc_auc_score(y_val, val_preds_ens)
    print(f"Ensemble Val ROC-AUC: {ens_auc:.4f}")

    print("Searching optimal threshold for Macro F_0.5...")
    thresholds = np.linspace(0.40, 0.95, 56)
    results = []
    for th in thresholds:
        preds = (val_preds_ens >= th).astype(int)
        p = precision_score(y_val, preds, zero_division=0)
        r = recall_score(y_val, preds, zero_division=0)
        f05 = compute_f_beta(p, r, beta=0.5)
        f1 = f1_score(y_val, preds, zero_division=0)
        results.append({'threshold': th, 'precision': p, 'recall': r, 'f05': f05, 'f1': f1})

    res_df = pd.DataFrame(results)
    best_idx = res_df['f05'].idxmax()
    best_row = res_df.loc[best_idx]
    optimal_th = float(best_row['threshold'])

    print(f"\n" + "=" * 50)
    print(f"OPTIMAL VALIDATION METRICS (13.1M Train / 2.31M Val)")
    print(f"=" * 50)
    print(f"Optimal Threshold (tau*): {optimal_th:.2f}")
    print(f"Validation Precision:     {best_row['precision']:.4f}")
    print(f"Validation Recall:        {best_row['recall']:.4f}")
    print(f"Validation Macro F_0.5:   {best_row['f05']:.4f}")
    print(f"Validation F_1:           {best_row['f1']:.4f}")
    print(f"=" * 50)

    # Save models locally
    os.makedirs('models', exist_ok=True)
    lgb_model_path = 'models/lightgbm_15m.txt'
    xgb_model_path = 'models/xgboost_15m.json'
    meta_path = 'models/ensemble_metadata.json'

    lgb_model.booster_.save_model(lgb_model_path)
    xgb_model.save_model(xgb_model_path)

    metadata = {
        'training_pairs': int(X.shape[0]),
        'train_samples': int(X_train.shape[0]),
        'val_samples': int(X_val.shape[0]),
        'optimal_threshold': round(optimal_th, 4),
        'val_precision': round(float(best_row['precision']), 4),
        'val_recall': round(float(best_row['recall']), 4),
        'val_f05': round(float(best_row['f05']), 4),
        'val_f1': round(float(best_row['f1']), 4),
        'lgb_auc': round(float(lgb_auc), 4),
        'xgb_auc': round(float(xgb_auc), 4),
        'ensemble_auc': round(float(ens_auc), 4),
        'lgb_train_time_sec': round(lgb_time, 2),
        'xgb_train_time_sec': round(xgb_time, 2),
        'features': [
            'name_ratio', 'name_token_sort', 'name_token_set', 'name_len_diff',
            'addr_ratio', 'addr_token_set', 'num_match', 'pin_match', 'is_s2'
        ]
    }
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
    print(f"\n[MODELS SAVED LOCALLY]")
    print(f"  - LightGBM: {lgb_model_path} ({os.path.getsize(lgb_model_path)/1024:.1f} KB)")
    print(f"  - XGBoost:  {xgb_model_path} ({os.path.getsize(xgb_model_path)/1024:.1f} KB)")
    print(f"  - Metadata: {meta_path}")

    # Free memory before test inference
    del X, y, X_train, X_val, y_train, y_val, val_preds_lgb, val_preds_xgb, val_preds_ens
    gc.collect()

    # 7. Test Candidate Scoring & Prediction Generation
    print("\n[STEP 6/6] Generating Test Predictions (matching_results.tsv)...")
    matching_out_path = 'output/matching_results.tsv'
    candidate_path = 'output/candidate_pairs.tsv'

    def extract_numbers(s):
        return set(re.findall(r'\b\d+\b', str(s)))

    print("Loading test source lookup tables...")
    t_inf_start = time.time()
    test_s1 = pd.read_csv('dataset/test/test_source1.tsv', sep='\t')
    test_s2 = pd.read_csv('dataset/test/test_source2.tsv', sep='\t')
    test_s3 = pd.read_csv('dataset/test/test_source3.tsv', sep='\t')

    target_lookup = {}
    for df_t in [test_s2, test_s3]:
        for tid, name, addr in zip(df_t['entity_id'], df_t['business_name'], df_t['business_address']):
            target_lookup[tid] = (str(name), str(addr))
    del test_s2, test_s3
    gc.collect()

    s1_lookup = {r['entity_id']: (str(r['business_name']), str(r['business_address'])) for _, r in test_s1.iterrows()}
    del test_s1
    gc.collect()
    print(f"Test tables ready in {time.time()-t_inf_start:.2f}s")

    print(f"Scoring 20.7M test candidate pairs (threshold = {optimal_th:.2f})...")
    batch_size = 40000
    matched_entities_count = 0
    singletons_count = 0
    total_processed = 0

    t_score_start = time.time()
    with open(candidate_path, 'r', encoding='utf-8') as in_f, \
         open(matching_out_path, 'w', encoding='utf-8', buffering=2*1024*1024) as out_f:

        out_f.write("source1_entity_id\tmatched_entity_ids\n")
        header = next(in_f)
        batch_records = []

        for line in in_f:
            s1_id, _, cand_str = line.partition('\t')
            cand_str = cand_str.strip()
            cands = [c.strip() for c in cand_str.split(',') if c.strip()] if cand_str else []
            batch_records.append((s1_id, cands))

            if len(batch_records) >= batch_size:
                pair_feats = []
                s1_slices = []
                cands_flat = []

                for sid, c_list in batch_records:
                    start_p = len(pair_feats)
                    s1_name, s1_addr = s1_lookup.get(sid, ('', ''))
                    nums_a = extract_numbers(s1_addr)
                    pin_a = re.findall(r'\b\d{5,6}\b', s1_addr)

                    for cid in c_list:
                        if cid in target_lookup:
                            t_name, t_addr = target_lookup[cid]
                            nr = fuzz.ratio(s1_name, t_name)
                            nts = fuzz.token_sort_ratio(s1_name, t_name)
                            ntset = fuzz.token_set_ratio(s1_name, t_name)
                            nld = abs(len(s1_name) - len(t_name))
                            ar = fuzz.ratio(s1_addr, t_addr)
                            atset = fuzz.token_set_ratio(s1_addr, t_addr)

                            nums_b = extract_numbers(t_addr)
                            nm = 1.0 if (nums_a and nums_b and (nums_a & nums_b)) else (0.5 if not nums_a or not nums_b else 0.0)

                            pin_b = re.findall(r'\b\d{5,6}\b', t_addr)
                            pm = 1.0 if (pin_a and pin_b and pin_a[0] == pin_b[0]) else (0.5 if not pin_a or not pin_b else 0.0)
                            is_s2 = 1.0 if cid.startswith('S2') else 0.0

                            pair_feats.append([nr, nts, ntset, nld, ar, atset, nm, pm, is_s2])
                            cands_flat.append(cid)

                    end_p = len(pair_feats)
                    s1_slices.append((sid, start_p, end_p))

                if pair_feats:
                    X_batch = np.array(pair_feats, dtype=np.float32)
                    scores = 0.50 * lgb_model.predict_proba(X_batch)[:, 1] + 0.50 * xgb_model.predict_proba(X_batch)[:, 1]
                else:
                    scores = np.array([])

                out_lines = []
                for sid, start_p, end_p in s1_slices:
                    if start_p == end_p:
                        singletons_count += 1
                        out_lines.append(f"{sid}\t\n")
                    else:
                        c_slice = cands_flat[start_p:end_p]
                        s_slice = scores[start_p:end_p]
                        matched = [c for c, sc in zip(c_slice, s_slice) if sc >= optimal_th]
                        if matched:
                            matched_entities_count += 1
                            out_lines.append(f"{sid}\t{','.join(matched)}\n")
                        else:
                            singletons_count += 1
                            out_lines.append(f"{sid}\t\n")

                out_f.writelines(out_lines)
                total_processed += len(batch_records)
                batch_records = []

                if total_processed % 400000 == 0:
                    dt = time.time() - t_score_start
                    pct = total_processed / 1732544 * 100
                    print(f"Processed {total_processed:,} / 1,732,544 ({pct:.1f}%) in {dt:.1f}s ({total_processed/dt:.0f} S1/sec)")

        # Remainder
        if batch_records:
            pair_feats = []
            s1_slices = []
            cands_flat = []
            for sid, c_list in batch_records:
                start_p = len(pair_feats)
                s1_name, s1_addr = s1_lookup.get(sid, ('', ''))
                nums_a = extract_numbers(s1_addr)
                pin_a = re.findall(r'\b\d{5,6}\b', s1_addr)
                for cid in c_list:
                    if cid in target_lookup:
                        t_name, t_addr = target_lookup[cid]
                        nr = fuzz.ratio(s1_name, t_name)
                        nts = fuzz.token_sort_ratio(s1_name, t_name)
                        ntset = fuzz.token_set_ratio(s1_name, t_name)
                        nld = abs(len(s1_name) - len(t_name))
                        ar = fuzz.ratio(s1_addr, t_addr)
                        atset = fuzz.token_set_ratio(s1_addr, t_addr)
                        nums_b = extract_numbers(t_addr)
                        nm = 1.0 if (nums_a and nums_b and (nums_a & nums_b)) else (0.5 if not nums_a or not nums_b else 0.0)
                        pin_b = re.findall(r'\b\d{5,6}\b', t_addr)
                        pm = 1.0 if (pin_a and pin_b and pin_a[0] == pin_b[0]) else (0.5 if not pin_a or not pin_b else 0.0)
                        is_s2 = 1.0 if cid.startswith('S2') else 0.0
                        pair_feats.append([nr, nts, ntset, nld, ar, atset, nm, pm, is_s2])
                        cands_flat.append(cid)
                end_p = len(pair_feats)
                s1_slices.append((sid, start_p, end_p))

            if pair_feats:
                X_batch = np.array(pair_feats, dtype=np.float32)
                scores = 0.50 * lgb_model.predict_proba(X_batch)[:, 1] + 0.50 * xgb_model.predict_proba(X_batch)[:, 1]
            else:
                scores = np.array([])

            out_lines = []
            for sid, start_p, end_p in s1_slices:
                if start_p == end_p:
                    singletons_count += 1
                    out_lines.append(f"{sid}\t\n")
                else:
                    c_slice = cands_flat[start_p:end_p]
                    s_slice = scores[start_p:end_p]
                    matched = [c for c, sc in zip(c_slice, s_slice) if sc >= optimal_th]
                    if matched:
                        matched_entities_count += 1
                        out_lines.append(f"{sid}\t{','.join(matched)}\n")
                    else:
                        singletons_count += 1
                        out_lines.append(f"{sid}\t\n")
            out_f.writelines(out_lines)
            total_processed += len(batch_records)

    total_inf_time = time.time() - t_score_start
    print(f"\nInference completed in {total_inf_time:.2f}s!")
    print(f"Total S1 entities processed:    {total_processed:,}")
    print(f"Total S1 entities with matches: {matched_entities_count:,} ({matched_entities_count/total_processed*100:.1f}%)")
    print(f"Predicted Singletons (no match): {singletons_count:,} ({singletons_count/total_processed*100:.1f}%)")
    print(f"Output saved to: {matching_out_path} ({os.path.getsize(matching_out_path)/1e6:.1f} MB)")

    # 8. Run validation
    print("\n[STEP 7/7] Running official validator check...")
    val_cmd = [
        "python3", "utils/validate_submission.py",
        "--matching", matching_out_path,
        "--candidate", candidate_path,
        "--test-dir", "dataset/test"
    ]
    val_res = subprocess.run(val_cmd, capture_output=True, text=True)
    print(val_res.stdout)
    if val_res.stderr:
        print("Stderr:", val_res.stderr)
    print(f"Validator Return Code: {val_res.returncode} ({'PASS' if val_res.returncode == 0 else 'FAIL'})")
    print("=" * 70)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 70)

if __name__ == '__main__':
    main()
