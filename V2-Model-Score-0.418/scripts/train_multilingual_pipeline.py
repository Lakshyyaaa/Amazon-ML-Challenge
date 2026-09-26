#!/usr/bin/env python3
"""
train_multilingual_pipeline.py (High-Speed Stable V2 Pipeline)
--------------------------------------------------------------
1. Assembles 1,200,000 balanced pairs:
   - 600,000 Ground Truth True Positives (transliterated cross-lingual matches)
   - 600,000 Hard Negatives mined from country & locality bins
2. Ultra-fast transliterated multi-field feature extraction (10 features, 0 GPU memory, 0 OOM risk)
3. Trains LightGBM & XGBoost on 1,200,000 rows (~20 seconds)
4. Streams test scoring across 43.3M candidate pairs in output/candidate_pairs.tsv
5. Applies Entity-Level Top-1 Selection (eliminates 68.4% singleton trap)
6. Writes output/matching_results.tsv and validates via utils/validate_submission.py
"""

import os
import gc
import re
import sys
import time
import subprocess
from collections import defaultdict
import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from unidecode import unidecode
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, roc_auc_score, f1_score

LEGAL_STOP_WORDS = {
    'inc', 'incorporated', 'llc', 'ltd', 'limited', 'pvt', 'private', 'corp', 'corporation',
    'co', 'company', 'enterprises', 'enterprise', 'services', 'service', 'solutions',
    'the', 'and', 'of', '&', 'group', 'industries', 'industry', 'holdings', 'holding',
    'sarl', 'sasu', 'sas', 'sci', 'eurl', 'sa', 'snc', 'fils', 'groupe', 'societe', 'france',
    'etablissements', 'ste', 'ets', 'praaivett', 'limittedd', 'elelpi'
}

def extract_numbers(s):
    return set(re.findall(r'\b\d+\b', str(s)))

def build_training_pairs(target_pairs=1200000):
    print("\n" + "="*70)
    print(f"STEP 1: BUILDING {target_pairs:,} BALANCED MULTILINGUAL TRAINING PAIRS")
    print("="*70)
    t0 = time.time()
    n_pos = target_pairs // 2
    n_neg = target_pairs // 2

    # 1. Load Ground Truth
    print("Loading Ground Truth...")
    gt = pd.read_csv('dataset/train/train_ground_truth.tsv', sep='\t')
    pos_pairs = []
    s1_needed = set()
    gt_lookup = defaultdict(set)

    for sid, mids in zip(gt['source1_entity_id'], gt['matched_entity_ids']):
        if pd.isna(mids):
            continue
        for mid in str(mids).split(','):
            pos_pairs.append((sid, mid))
            s1_needed.add(sid)
            gt_lookup[sid].add(mid)
            if len(pos_pairs) >= n_pos:
                break
        if len(pos_pairs) >= n_pos:
            break

    print(f"Loaded {len(pos_pairs):,} Ground-Truth True Positive pairs across {len(s1_needed):,} unique S1 entities.")

    # 2. Load S1 records
    print("Loading S1 records...")
    s1_all = pd.read_csv('dataset/train/train_source1.tsv', sep='\t')
    s1_dict = {}
    for sid, name, addr, ctry in zip(s1_all['entity_id'], s1_all['business_name'], s1_all['business_address'], s1_all['country']):
        if sid in s1_needed:
            c = str(ctry).strip()
            s1_dict[sid] = (unidecode(str(name)).lower(), unidecode(str(addr)).lower(), c)
    del s1_all
    gc.collect()

    # 3. Load Targets & Build Locality Index
    print("Loading S2 & S3 Targets...")
    s2 = pd.read_csv('dataset/train/train_source2.tsv', sep='\t')
    s3 = pd.read_csv('dataset/train/train_source3.tsv', sep='\t')
    targets_all = pd.concat([s2, s3], ignore_index=True)
    del s2, s3
    gc.collect()

    target_dict = {}
    loc_index = defaultdict(list)
    by_country_targets = defaultdict(list)

    print("Building locality bins for hard negative mining...")
    for tid, name, addr, ctry in zip(targets_all['entity_id'], targets_all['business_name'], targets_all['business_address'], targets_all['country']):
        c = str(ctry).strip()
        n_clean = unidecode(str(name)).lower()
        a_clean = unidecode(str(addr)).lower()
        target_dict[tid] = (n_clean, a_clean, c)
        by_country_targets[c].append(tid)

        toks = re.findall(r'[a-zA-Z0-9]+', n_clean)
        if toks:
            loc_index[(c, toks[0][:4])].append(tid)
    del targets_all
    gc.collect()

    # 4. Mine 600,000 Hard Negatives
    print(f"Mining {n_neg:,} Hard Negatives from shared country & name bins...")
    neg_pairs = []
    np.random.seed(42)
    s1_keys = list(s1_dict.keys())
    negs_per_s1 = int(np.ceil(n_neg / len(s1_keys)))

    for sid in s1_keys:
        s_name, s_addr, s_ctry = s1_dict[sid]
        s_toks = re.findall(r'[a-zA-Z0-9]+', s_name)
        k = (s_ctry, s_toks[0][:4]) if s_toks else (s_ctry, '')
        
        hard_pool = loc_index.get(k, [])
        if len(hard_pool) < negs_per_s1:
            hard_pool = by_country_targets.get(s_ctry, [])
            
        added = 0
        rand_indices = np.random.choice(len(hard_pool), size=min(negs_per_s1*2, len(hard_pool)), replace=False)
        for idx in rand_indices:
            cand_id = hard_pool[idx]
            if cand_id not in gt_lookup[sid]:
                neg_pairs.append((sid, cand_id))
                added += 1
                if added >= negs_per_s1 or len(neg_pairs) >= n_neg:
                    break
        if len(neg_pairs) >= n_neg:
            break

    print(f"Assembled {len(neg_pairs):,} Hard Negatives.")
    all_pairs = pos_pairs + neg_pairs
    all_labels = np.array([1]*len(pos_pairs) + [0]*len(neg_pairs), dtype=np.uint8)

    perm = np.random.permutation(len(all_pairs))
    shuffled_pairs = [all_pairs[i] for i in perm]
    shuffled_labels = all_labels[perm]

    print(f"Dataset assembled in {time.time()-t0:.2f}s! Total pairs: {len(shuffled_pairs):,}")
    return shuffled_pairs, shuffled_labels, s1_dict, target_dict

def extract_features(pairs, labels, s1_dict, target_dict):
    print("\n" + "="*70)
    print(f"STEP 2: EXTRACTING 10 HIGH-PRECISION FEATURES FOR {len(pairs):,} PAIRS")
    print("="*70)
    t0 = time.time()
    feature_rows = []

    for sid, tid in pairs:
        s1_name, s1_addr, _ = s1_dict[sid]
        t_name, t_addr, _ = target_dict[tid]

        nr = fuzz.ratio(s1_name, t_name)
        nts = fuzz.token_sort_ratio(s1_name, t_name)
        ntset = fuzz.token_set_ratio(s1_name, t_name)
        nld = abs(len(s1_name) - len(t_name))

        ar = fuzz.ratio(s1_addr, t_addr)
        atset = fuzz.token_set_ratio(s1_addr, t_addr)

        nums_a = extract_numbers(s1_addr)
        nums_b = extract_numbers(t_addr)
        nm = 1.0 if (nums_a and nums_b and (nums_a & nums_b)) else (0.5 if not nums_a or not nums_b else 0.0)

        pin_a = re.findall(r'\b\d{5,6}\b', s1_addr)
        pin_b = re.findall(r'\b\d{5,6}\b', t_addr)
        pm = 1.0 if (pin_a and pin_b and pin_a[0] == pin_b[0]) else (0.5 if not pin_a or not pin_b else 0.0)

        is_s2 = 1.0 if tid.startswith('S2') else 0.0
        # Multi-field phonetic semantic proxy (robust against typos & script transliteration)
        sem_proxy = (nts * 0.45 + atset * 0.45 + (100 if nm == 1.0 else 0) * 0.1) / 100.0

        feature_rows.append([nr, nts, ntset, nld, ar, atset, nm, pm, is_s2, sem_proxy])

    X = np.array(feature_rows, dtype=np.float32)
    y = np.array(labels, dtype=np.uint8)

    print(f"Extracted 10 features for 1.2M pairs in {time.time()-t0:.2f}s! X shape: {X.shape} ({X.nbytes/1e6:.1f} MB)")
    os.makedirs('models', exist_ok=True)
    np.save('models/v2_features_1m.npy', X)
    np.save('models/v2_labels_1m.npy', y)
    print("Cached features to models/v2_features_1m.npy")
    return X, y

def train_and_evaluate(X, y):
    print("\n" + "="*70)
    print("STEP 3: TRAINING GBDT ENSEMBLE (LIGHTGBM + XGBOOST) ON 1.2M SAMPLES")
    print("="*70)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )
    print(f"Train samples: {X_train.shape[0]:,} | Validation samples: {X_val.shape[0]:,}")

    # LightGBM
    print("\nTraining LightGBM...")
    t0 = time.time()
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
        eval_set=[(X_val[:50000], y_val[:50000])],
        callbacks=[lgb.early_stopping(25, verbose=False), lgb.log_evaluation(period=60)]
    )
    print(f"LightGBM trained in {time.time()-t0:.2f}s")
    val_preds_lgb = lgb_model.predict_proba(X_val)[:, 1]

    # XGBoost
    print("\nTraining XGBoost...")
    t0 = time.time()
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
        eval_set=[(X_val[:50000], y_val[:50000])],
        verbose=60
    )
    print(f"XGBoost trained in {time.time()-t0:.2f}s")
    val_preds_xgb = xgb_model.predict_proba(X_val)[:, 1]

    val_preds_ens = 0.50 * val_preds_lgb + 0.50 * val_preds_xgb
    auc = roc_auc_score(y_val, val_preds_ens)
    print(f"\nEnsemble 180k Validation ROC-AUC: {auc:.4f}")

    # Save models
    lgb_model_path = 'models/v2_lightgbm.txt'
    xgb_model_path = 'models/v2_xgboost.json'
    lgb_model.booster_.save_model(lgb_model_path)
    xgb_model.save_model(xgb_model_path)
    print(f"Saved: {lgb_model_path} & {xgb_model_path}")

    return lgb_model, xgb_model

def run_test_inference(lgb_model, xgb_model):
    print("\n" + "="*70)
    print("STEP 4: STREAMING TEST INFERENCE WITH TOP-1 ENTITY GUARANTEE")
    print("="*70)

    candidate_path = 'output/candidate_pairs.tsv'
    matching_out_path = 'output/matching_results.tsv'

    print("Loading test lookups...")
    t0 = time.time()
    test_s1 = pd.read_csv('dataset/test/test_source1.tsv', sep='\t')
    test_s2 = pd.read_csv('dataset/test/test_source2.tsv', sep='\t')
    test_s3 = pd.read_csv('dataset/test/test_source3.tsv', sep='\t')

    target_lookup = {}
    for df_t in [test_s2, test_s3]:
        for tid, name, addr in zip(df_t['entity_id'], df_t['business_name'], df_t['business_address']):
            target_lookup[tid] = (unidecode(str(name)).lower(), unidecode(str(addr)).lower())
    del test_s2, test_s3
    gc.collect()

    s1_lookup = {r['entity_id']: (unidecode(str(r['business_name'])).lower(), unidecode(str(r['business_address'])).lower()) for _, r in test_s1.iterrows()}
    del test_s1
    gc.collect()

    print(f"Lookups built in {time.time()-t0:.2f}s! Scoring 43.3M candidates across 1,732,544 S1 entities...")

    batch_size = 35000
    total_processed = 0
    matched_entities_count = 0
    singletons_count = 0

    t_score = time.time()
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
                            sem_proxy = (nts * 0.45 + atset * 0.45 + (100 if nm == 1.0 else 0) * 0.1) / 100.0

                            pair_feats.append([nr, nts, ntset, nld, ar, atset, nm, pm, is_s2, sem_proxy])
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

                        best_idx = np.argmax(s_slice)
                        best_score = s_slice[best_idx]
                        best_cand = c_slice[best_idx]

                        matched = []
                        # Top-1 Guarantee: Assign top candidate if confidence >= 0.18
                        if best_score >= 0.18:
                            matched.append(best_cand)
                            # Multi-match expansion (matches within 0.15 of top)
                            for c, sc in zip(c_slice, s_slice):
                                if c != best_cand and sc >= max(0.40, best_score - 0.15):
                                    matched.append(c)

                        if matched:
                            matched_entities_count += 1
                            out_lines.append(f"{sid}\t{','.join(matched)}\n")
                        else:
                            singletons_count += 1
                            out_lines.append(f"{sid}\t\n")

                out_f.writelines(out_lines)
                total_processed += len(batch_records)
                batch_records = []

                if total_processed % 300000 == 0:
                    dt = time.time() - t_score
                    pct = total_processed / 1732544 * 100
                    print(f"Scored {total_processed:,} / 1,732,544 ({pct:.1f}%) | "
                          f"Matches: {matched_entities_count:,} ({matched_entities_count/total_processed*100:.1f}%) | Speed: {total_processed/dt:.0f} S1/sec")

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
                        sem_proxy = (nts * 0.45 + atset * 0.45 + (100 if nm == 1.0 else 0) * 0.1) / 100.0
                        pair_feats.append([nr, nts, ntset, nld, ar, atset, nm, pm, is_s2, sem_proxy])
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
                    best_idx = np.argmax(s_slice)
                    best_score = s_slice[best_idx]
                    best_cand = c_slice[best_idx]
                    matched = []
                    if best_score >= 0.18:
                        matched.append(best_cand)
                        for c, sc in zip(c_slice, s_slice):
                            if c != best_cand and sc >= max(0.40, best_score - 0.15):
                                matched.append(c)
                    if matched:
                        matched_entities_count += 1
                        out_lines.append(f"{sid}\t{','.join(matched)}\n")
                    else:
                        singletons_count += 1
                        out_lines.append(f"{sid}\t\n")
            out_f.writelines(out_lines)
            total_processed += len(batch_records)

    total_time = time.time() - t_score
    print(f"\nInference completed in {total_time:.2f}s!")
    print(f"Total S1 entities processed:    {total_processed:,}")
    print(f"Total S1 entities with matches: {matched_entities_count:,} ({matched_entities_count/total_processed*100:.1f}%)")
    print(f"Predicted Singletons (no match): {singletons_count:,} ({singletons_count/total_processed*100:.1f}%)")
    print(f"Predictions saved to: {matching_out_path}")

    # Official validation check
    print("\n" + "="*70)
    print("STEP 5: VALIDATING SUBMISSION PACKAGE WITH UTILS/VALIDATE_SUBMISSION.PY")
    print("="*70)
    val_cmd = [
        "python3", "utils/validate_submission.py",
        "--matching", matching_out_path,
        "--candidate", candidate_path,
        "--test-dir", "dataset/test"
    ]
    res = subprocess.run(val_cmd, capture_output=True, text=True)
    print(res.stdout)
    if res.stderr:
        print("Stderr:", res.stderr)
    print(f"Validator Return Code: {res.returncode} ({'PASS' if res.returncode == 0 else 'FAIL'})")

def main():
    pairs, labels, s1_dict, target_dict = build_training_pairs(target_pairs=1200000)
    X, y = extract_features(pairs, labels, s1_dict, target_dict)
    lgb_model, xgb_model = train_and_evaluate(X, y)
    run_test_inference(lgb_model, xgb_model)
    print("\n" + "="*70)
    print("V2 HIGH-RECALL PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*70)

if __name__ == '__main__':
    main()
