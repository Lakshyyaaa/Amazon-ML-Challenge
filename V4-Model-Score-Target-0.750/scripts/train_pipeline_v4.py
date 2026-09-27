#!/usr/bin/env python3
"""
train_pipeline_v4.py (V4 Training Engine with 24-D Structural & Spatial Features)
--------------------------------------------------------------------------------
1. Mines 1,000,000 Ground Truth Positives + 1,000,000 Real 10-Route Blocker Hard Negatives.
2. Extracts 24 fine-grained features:
   - Name & Phonetics: ratio, token set, token sort, partial, len diff, soundex
   - Address Components: addr ratio, addr token set, exact street match, num exact,
     num zero-padded, num suffix-stripped, PIN match, locality match
   - Trade Alias Spatial Flag: addr_fuzz * (1.0 - name_fuzz)
   - Multilingual & Indic: cross-script transliteration ratio
   - Domain & URL match: domain token ratio
   - Source flag: is_s2
3. Trains regularized 3-Booster ensemble (LightGBM, XGBoost, CatBoost).
4. Saves models & configuration to V4-Model-Score-Target-0.950/models/.
"""

import os
import re
import gc
import sys
import time
import json
from collections import defaultdict
import numpy as np
import pandas as pd
from unidecode import unidecode
from rapidfuzz import fuzz
import jellyfish
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_score, recall_score
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from generate_candidates_v4 import extract_v4_keys, detect_and_transliterate

STREET_NOISE = {
    'road', 'street', 'avenue', 'drive', 'lane', 'court', 'way', 'rue', 'boulevard',
    'allee', 'st', 'rd', 'ave', 'dr', 'blvd', 'township', 'block', 'plot', 'near',
    'floor', 'flat', 'phase', 'sector', 'nagar', 'colony', 'urban', 'rural', 'post', 'box'
}

def extract_clean_numbers(text):
    if not text:
        return []
    # Strip leading zeros and alpha suffixes
    raw_nums = re.findall(r'\b0*(\d+)[a-zA-Z]?\b', str(text))
    return raw_nums

def compute_24d_features(s_name, s_addr, t_name, t_addr, s_trans_name, t_trans_name, is_s2):
    # 1. Standard Lexical Name
    nf = fuzz.ratio(s_name, t_name) / 100.0
    ntset = fuzz.token_set_ratio(s_name, t_name) / 100.0
    ntsort = fuzz.token_sort_ratio(s_name, t_name) / 100.0
    npart = fuzz.partial_ratio(s_name, t_name) / 100.0
    nld = abs(len(s_name) - len(t_name))
    
    # 2. Phonetic Soundex & Metaphone match
    s_toks = [t for t in re.findall(r'[a-zA-Z]+', s_name) if len(t) >= 3]
    t_toks = [t for t in re.findall(r'[a-zA-Z]+', t_name) if len(t) >= 3]
    snd_m = 1.0 if (s_toks and t_toks and jellyfish.soundex(s_toks[0]) == jellyfish.soundex(t_toks[0])) else 0.0
    met_m = 1.0 if (s_toks and t_toks and jellyfish.metaphone(s_toks[0]) == jellyfish.metaphone(t_toks[0])) else 0.0

    # 3. Transliterated Indic Cross-Script Ratio
    trans_ratio = fuzz.token_set_ratio(s_trans_name, t_trans_name) / 100.0

    # 4. Domain & Handle extraction
    s_dom = re.search(r'\b([a-zA-Z0-9]+)\.(?:com|org|net|in|fr)\b', s_name)
    t_dom = re.search(r'\b([a-zA-Z0-9]+)\.(?:com|org|net|in|fr)\b', t_name)
    dom_ratio = 1.0 if (s_dom and t_dom and s_dom.group(1) == t_dom.group(1)) else 0.0

    # 5. Address Lexical
    af = fuzz.ratio(s_addr, t_addr) / 100.0
    atset = fuzz.token_set_ratio(s_addr, t_addr) / 100.0

    # 6. Building Number Hierarchy
    s_nums_raw = re.findall(r'\b\d+\b', s_addr)
    t_nums_raw = re.findall(r'\b\d+\b', t_addr)
    num_exact = 1.0 if (s_nums_raw and t_nums_raw and (set(s_nums_raw) & set(t_nums_raw))) else 0.0

    s_clean_nums = extract_clean_numbers(s_addr)
    t_clean_nums = extract_clean_numbers(t_addr)
    num_norm = 1.0 if (s_clean_nums and t_clean_nums and (set(s_clean_nums) & set(t_clean_nums))) else 0.0

    # 7. Postal/PIN Code Match
    pins_a = re.findall(r'\b\d{5,6}\b', s_addr)
    pins_b = re.findall(r'\b\d{5,6}\b', t_addr)
    pin_m = 1.0 if (pins_a and pins_b and pins_a[0] == pins_b[0]) else 0.0

    # 8. Street & Locality Words
    s_words = set(w for w in re.findall(r'[a-zA-Z]{4,}', s_addr) if w not in STREET_NOISE)
    t_words = set(w for w in re.findall(r'[a-zA-Z]{4,}', t_addr) if w not in STREET_NOISE)
    street_m = len(s_words & t_words) / max(1, len(s_words | t_words))

    # 9. Spatial Trade Alias Interaction Flag (co-located DBA flag)
    spatial_alias_flag = atset * (1.0 - ntset)

    # 10. Additional signals
    len_ratio = min(len(s_name), len(t_name)) / max(1, max(len(s_name), len(t_name)))
    both_have_pin = 1.0 if (pins_a and pins_b) else 0.0
    pin_mismatch = 1.0 if (pins_a and pins_b and pins_a[0] != pins_b[0]) else 0.0

    return [
        nf, ntset, ntsort, npart, nld, snd_m, met_m, trans_ratio, dom_ratio,
        af, atset, num_exact, num_norm, pin_m, street_m, spatial_alias_flag,
        len_ratio, both_have_pin, pin_mismatch, is_s2
    ]

def main():
    print("=" * 75)
    print("V4 MODEL TRAINING PIPELINE (20-D SPATIAL & INDIC FEATURE MATRIX)")
    print("=" * 75)
    t0 = time.time()
    n_pos_target = 1000000
    n_neg_target = 1000000

    # 1. Load Ground Truth Positives
    print("\n[STEP 1/5] Loading 1,000,000 Ground Truth Positives...")
    gt_path = 'dataset/train/train_ground_truth.tsv'
    gt_map = defaultdict(set)
    pos_pairs = []
    s1_needed = set()
    needed_targets = set()

    for chunk in pd.read_csv(gt_path, sep='\t', chunksize=100000):
        for sid, mids in zip(chunk['source1_entity_id'], chunk['matched_entity_ids']):
            if pd.isna(mids):
                continue
            for mid in str(mids).split(','):
                pos_pairs.append((sid, mid))
                gt_map[sid].add(mid)
                needed_targets.add(mid)
                s1_needed.add(sid)
                if len(pos_pairs) >= n_pos_target:
                    break
            if len(pos_pairs) >= n_pos_target:
                break
        if len(pos_pairs) >= n_pos_target:
            break

    print(f"Loaded {len(pos_pairs):,} Ground-Truth True Positive pairs across {len(s1_needed):,} S1 entities.")

    # 2. Load S1 Query Data
    print(f"\n[STEP 2/5] Loading S1 metadata for {len(s1_needed):,} query entities...")
    s1_dict = {}
    for chunk in pd.read_csv('dataset/train/train_source1.tsv', sep='\t', chunksize=200000):
        sub = chunk[chunk['entity_id'].isin(s1_needed)]
        for sid, name, addr, ctry in zip(sub['entity_id'], sub['business_name'], sub['business_address'], sub['country']):
            n_raw = str(name).strip() if pd.notna(name) else ""
            n_trans = unidecode(detect_and_transliterate(n_raw)).lower()
            n_clean = unidecode(n_raw).lower()
            a_clean = unidecode(str(addr)).lower() if pd.notna(addr) else ""
            s1_dict[sid] = (n_clean, a_clean, str(ctry).strip(), n_trans, n_raw)
        if len(s1_dict) >= len(s1_needed):
            break

    print(f"Loaded {len(s1_dict):,} S1 query entities into memory.")

    # 3. Load & Index Target Entities for 10-Route Blocker Mining
    print("\n[STEP 3/5] Indexing target entities into V4 10-Route Inverted Index...")
    t_idx = time.time()
    target_dict = {}
    index = defaultdict(list)
    total_targets_indexed = 0
    max_distractor_targets = 2500000

    for src_path, is_s2_flag in [('dataset/train/train_source2.tsv', 1.0), ('dataset/train/train_source3.tsv', 0.0)]:
        print(f"  Indexing from {src_path}...")
        for chunk in pd.read_csv(src_path, sep='\t', chunksize=250000):
            for tid, name, addr, ctry in zip(chunk['entity_id'], chunk['business_name'], chunk['business_address'], chunk['country']):
                if tid in needed_targets or total_targets_indexed < max_distractor_targets:
                    n_raw = str(name).strip() if pd.notna(name) else ""
                    n_trans = unidecode(detect_and_transliterate(n_raw)).lower()
                    n_clean = unidecode(n_raw).lower()
                    a_clean = unidecode(str(addr)).lower() if pd.notna(addr) else ""
                    c_clean = str(ctry).strip()
                    target_dict[tid] = (n_clean, a_clean, c_clean, is_s2_flag, n_trans, n_raw)
                    total_targets_indexed += 1
                    for k in extract_v4_keys(n_raw, a_clean, c_clean):
                        index[k].append(tid)

    print(f"Indexed {len(target_dict):,} targets across {len(index):,} unique bins in {time.time()-t_idx:.2f}s.")

    # 4. Mine 1,000,000 10-Route Blocker Hard Negatives
    print(f"\n[STEP 4/5] Mining {n_neg_target:,} 10-ROUTE BLOCKER HARD NEGATIVES...")
    t_mine = time.time()
    neg_pairs = []
    negs_per_s1 = int(np.ceil(n_neg_target / len(s1_dict)))

    for s1_idx, (sid, (s_name, s_addr, s_ctry, s_trans, s_raw)) in enumerate(s1_dict.items(), 1):
        cand_scores = defaultdict(float)
        for k in extract_v4_keys(s_raw, s_addr, s_ctry):
            bin_recs = index.get(k, [])
            n_bin = len(bin_recs)
            if n_bin > 0:
                w = 1.0 / (1.0 + n_bin / 100.0)
                for tid in bin_recs[:100]:
                    cand_scores[tid] += w

        sorted_blocker_cands = sorted(cand_scores.items(), key=lambda x: x[1], reverse=True)[:35]
        true_set = gt_map[sid]
        added = 0
        for tid, score in sorted_blocker_cands:
            if tid not in true_set and tid in target_dict:
                neg_pairs.append((sid, tid))
                added += 1
                if added >= negs_per_s1 or len(neg_pairs) >= n_neg_target:
                    break
        if len(neg_pairs) >= n_neg_target:
            break

        if s1_idx % 50000 == 0:
            print(f"  Processed {s1_idx:,}/{len(s1_dict):,} S1 | Mined {len(neg_pairs):,} Blocker Hard Negatives ({len(neg_pairs)/n_neg_target*100:.1f}%)")

    print(f"Successfully mined {len(neg_pairs):,} 10-ROUTE BLOCKER HARD NEGATIVES in {time.time()-t_mine:.2f}s!")
    del index
    gc.collect()

    # 5. Extract 20-D Structural & Spatial Features
    print("\n[STEP 5/5] Extracting 20-dimensional structural features for 2,000,000 pairs...")
    t_feat = time.time()
    all_pairs = pos_pairs[:n_pos_target] + neg_pairs[:n_neg_target]
    all_labels = np.array([1]*len(pos_pairs[:n_pos_target]) + [0]*len(neg_pairs[:n_neg_target]), dtype=np.uint8)

    perm = np.random.RandomState(42).permutation(len(all_pairs))
    shuffled_pairs = [all_pairs[i] for i in perm]
    y_all = all_labels[perm]

    X_mat = np.zeros((len(shuffled_pairs), 20), dtype=np.float32)

    for i, (sid, tid) in enumerate(shuffled_pairs):
        s_name, s_addr, _, _, s_trans = s1_dict[sid]
        t_name, t_addr, _, is_s2, t_trans, _ = target_dict[tid]

        feats = compute_24d_features(s_name, s_addr, t_name, t_addr, s_trans, t_trans, is_s2)
        X_mat[i] = feats

        if (i + 1) % 500000 == 0:
            print(f"  Extracted {i+1:,} / {len(shuffled_pairs):,} pairs ({((i+1)/len(shuffled_pairs))*100:.1f}%)")

    print(f"Feature matrix built in {time.time()-t_feat:.2f}s: X {X_mat.shape} ({X_mat.nbytes/1e6:.1f} MB), y {y_all.shape}")

    # Cache features to V4 folder
    feat_cache_dir = 'V4-Model-Score-Target-0.950/features'
    os.makedirs(feat_cache_dir, exist_ok=True)
    f_path = os.path.join(feat_cache_dir, 'v4_train_features_2m.npy')
    l_path = os.path.join(feat_cache_dir, 'v4_train_labels_2m.npy')
    np.save(f_path, X_mat)
    np.save(l_path, y_all)
    print(f"Saved cached binary features: {f_path} ({os.path.getsize(f_path)/1e6:.1f} MB)")
    print(f"Saved cached binary labels:   {l_path} ({os.path.getsize(l_path)/1e6:.1f} MB)")

    # 6. Train 3-Booster Ensemble
    print("\n" + "=" * 75)
    print("TRAINING REGULARIZED 3-BOOSTER ENSEMBLE ON V4 20-D FEATURES")
    print("=" * 75)
    X_train, X_val, y_train, y_val = train_test_split(X_mat, y_all, test_size=0.15, random_state=42, stratify=y_all)

    model_dir = 'V4-Model-Score-Target-0.950/models'
    os.makedirs(model_dir, exist_ok=True)

    # LightGBM
    print("\nTraining LightGBM (250 trees, max_depth=6, L1=2.0, L2=5.0)...")
    lgb_train = lgb.Dataset(X_train, label=y_train)
    lgb_val = lgb.Dataset(X_val, label=y_val, reference=lgb_train)
    lgb_params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'learning_rate': 0.08,
        'num_leaves': 31,
        'max_depth': 6,
        'lambda_l1': 2.0,
        'lambda_l2': 5.0,
        'feature_fraction': 0.80,
        'bagging_fraction': 0.80,
        'bagging_freq': 1,
        'verbose': -1,
        'random_state': 42,
        'n_jobs': 4
    }
    t_lgb = time.time()
    lgb_model = lgb.train(lgb_params, lgb_train, num_boost_round=250, valid_sets=[lgb_val], callbacks=[lgb.log_evaluation(50)])
    p_val_lgb = lgb_model.predict(X_val)
    auc_lgb = roc_auc_score(y_val, p_val_lgb)
    print(f"LightGBM trained in {time.time()-t_lgb:.2f}s | Val AUC: {auc_lgb:.5f}")
    lgb_model.save_model(os.path.join(model_dir, 'v4_lightgbm.txt'))

    # XGBoost
    print("\nTraining XGBoost (250 trees, max_depth=5, L1=2.0, L2=5.0)...")
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    xgb_params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'learning_rate': 0.08,
        'max_depth': 5,
        'alpha': 2.0,
        'lambda': 5.0,
        'subsample': 0.80,
        'colsample_bytree': 0.80,
        'random_state': 42,
        'tree_method': 'hist',
        'n_jobs': 4
    }
    t_xgb = time.time()
    xgb_model = xgb.train(xgb_params, dtrain, num_boost_round=250, evals=[(dval, 'val')], verbose_eval=50)
    p_val_xgb = xgb_model.predict(dval)
    auc_xgb = roc_auc_score(y_val, p_val_xgb)
    print(f"XGBoost trained in {time.time()-t_xgb:.2f}s | Val AUC: {auc_xgb:.5f}")
    xgb_model.save_model(os.path.join(model_dir, 'v4_xgboost.json'))

    # CatBoost
    print("\nTraining CatBoost (250 trees, depth=6, l2_leaf_reg=5.0)...")
    cat_model = CatBoostClassifier(
        iterations=250,
        learning_rate=0.08,
        depth=6,
        l2_leaf_reg=5.0,
        random_seed=42,
        verbose=50,
        thread_count=4
    )
    t_cat = time.time()
    cat_model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=50)
    p_val_cat = cat_model.predict_proba(X_val)[:, 1]
    auc_cat = roc_auc_score(y_val, p_val_cat)
    print(f"CatBoost trained in {time.time()-t_cat:.2f}s | Val AUC: {auc_cat:.5f}")
    cat_model.save_model(os.path.join(model_dir, 'v4_catboost.cbm'))

    # Ensemble Evaluation
    p_val_ens = (0.35 * p_val_lgb) + (0.35 * p_val_xgb) + (0.30 * p_val_cat)
    auc_ens = roc_auc_score(y_val, p_val_ens)
    print(f"\n3-Booster Blended Ensemble Val AUC: {auc_ens:.5f}")

    # Optimal Threshold Search
    best_th = 0.50
    best_f05 = 0.0
    best_p, best_r = 0.0, 0.0
    for th in np.linspace(0.40, 0.90, 51):
        b_pred = (p_val_ens >= th).astype(int)
        p = precision_score(y_val, b_pred, zero_division=0)
        r = recall_score(y_val, b_pred, zero_division=0)
        f05 = (1.25 * p * r) / (0.25 * p + r + 1e-10)
        if f05 > best_f05:
            best_f05 = f05
            best_th = th
            best_p = p
            best_r = r

    print(f"Optimal Decision Threshold: tau = {best_th:.3f}")
    print(f"  Precision: {best_p*100:.2f}% | Recall: {best_r*100:.2f}% | Macro F0.5: {best_f05:.4f}")

    # Save ensemble metadata
    config = {
        'models': ['v4_lightgbm.txt', 'v4_xgboost.json', 'v4_catboost.cbm'],
        'weights': {'lightgbm': 0.35, 'xgboost': 0.35, 'catboost': 0.30},
        'optimal_threshold': float(best_th),
        'val_auc': float(auc_ens),
        'val_f05': float(best_f05),
        'precision': float(best_p),
        'recall': float(best_r),
        'features': [
            'name_fuzz', 'name_token_set', 'name_token_sort', 'name_partial', 'name_len_diff',
            'soundex_match', 'metaphone_match', 'translit_ratio', 'domain_ratio',
            'addr_fuzz', 'addr_token_set', 'num_exact', 'num_norm', 'pin_match', 'street_match',
            'spatial_alias_flag', 'len_ratio', 'both_have_pin', 'pin_mismatch', 'is_s2'
        ]
    }
    cfg_path = os.path.join(model_dir, 'v4_ensemble_config.json')
    with open(cfg_path, 'w') as f:
        json.dump(config, f, indent=2)

    total_time = time.time() - t0
    print("\n" + "=" * 75)
    print(f"V4 TRAINING PIPELINE COMPLETE IN {total_time:.2f}s ({total_time/60:.1f} minutes)")
    print("=" * 75)

if __name__ == '__main__':
    main()
