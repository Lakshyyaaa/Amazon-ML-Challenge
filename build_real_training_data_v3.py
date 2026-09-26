#!/usr/bin/env python3
"""
build_real_training_data_v3.py
--------------------------------
1. Assembles exactly 2,000,000 genuine training pairs from dataset/train/:
   - 1,000,000 Ground-Truth True Positives from dataset/train/train_ground_truth.tsv
   - 1,000,000 Real Blocker Hard Negatives mined directly by running the V3 6-route
     blocker on the same S1 entities (top IDF-scoring non-ground-truth collisions).
2. Computes the 10 RapidFuzz lexical & address features in parallel/vectorized C.
3. Caches to output/v3_features/v3_train_features_2m.npy and v3_train_labels_2m.npy.
4. Trains the regularized 3-booster ensemble (LightGBM, XGBoost, CatBoost).
5. Saves models to V3-Model-Score-0.742/models/.
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
    if addr and pd.notna(addr):
        a_clean = unidecode(str(addr)).lower()
        pins = re.findall(r'\b\d{5,6}\b', a_clean)
        for p in pins[:2]:
            keys.append('pin_' + p)
        nums = re.findall(r'\b\d+\b', a_clean)
        for n in nums[:3]:
            keys.append('num_' + n)
        a_words = [t for t in re.findall(r'[a-zA-Z]{4,}', a_clean) if t not in STREET_NOISE]
        for w in a_words[:5]:
            keys.append('aword_' + w)
            if nums:
                keys.append(f'nw_{nums[0]}_{w}')
    return [(c, k) for k in keys]

def extract_numbers(text):
    if not text:
        return set()
    return set(re.findall(r'\b\d+\b', text))

def main():
    print("=" * 75)
    print("V3 REAL DATA ASSEMBLY: MINING ACTUAL BLOCKER HARD NEGATIVES & TRAINING")
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
            n_clean = unidecode(str(name)).lower() if pd.notna(name) else ""
            a_clean = unidecode(str(addr)).lower() if pd.notna(addr) else ""
            s1_dict[sid] = (n_clean, a_clean, str(ctry).strip())
        if len(s1_dict) >= len(s1_needed):
            break

    print(f"Loaded {len(s1_dict):,} S1 query entities into memory.")

    # 3. Load & Index Target Entities for Blocker Mining
    print("\n[STEP 3/5] Indexing target entities into V3 6-Route Inverted Index...")
    t_idx = time.time()
    target_dict = {}
    index = defaultdict(list)
    total_targets_indexed = 0
    max_distractor_targets = 2000000 # 2M distractor pool + all needed true targets

    for src_path, is_s2_flag in [('dataset/train/train_source2.tsv', 1.0), ('dataset/train/train_source3.tsv', 0.0)]:
        print(f"  Indexing from {src_path}...")
        for chunk in pd.read_csv(src_path, sep='\t', chunksize=250000):
            for tid, name, addr, ctry in zip(chunk['entity_id'], chunk['business_name'], chunk['business_address'], chunk['country']):
                # Index if it's a needed target OR if we still have room in distractor pool
                if tid in needed_targets or total_targets_indexed < max_distractor_targets:
                    n_clean = unidecode(str(name)).lower() if pd.notna(name) else ""
                    a_clean = unidecode(str(addr)).lower() if pd.notna(addr) else ""
                    c_clean = str(ctry).strip()
                    target_dict[tid] = (n_clean, a_clean, c_clean, is_s2_flag)
                    total_targets_indexed += 1
                    for k in extract_v3_keys(n_clean, a_clean, c_clean):
                        index[k].append(tid)

    print(f"Indexed {len(target_dict):,} targets across {len(index):,} unique bins in {time.time()-t_idx:.2f}s.")

    # 4. Mine 1,000,000 Actual Blocker Hard Negatives
    print(f"\n[STEP 4/5] Mining {n_neg_target:,} ACTUAL BLOCKER HARD NEGATIVES...")
    t_mine = time.time()
    neg_pairs = []
    negs_per_s1 = int(np.ceil(n_neg_target / len(s1_dict)))

    for s1_idx, (sid, (s_name, s_addr, s_ctry)) in enumerate(s1_dict.items(), 1):
        cand_scores = defaultdict(float)
        for k in extract_v3_keys(s_name, s_addr, s_ctry):
            bin_recs = index.get(k, [])
            n_bin = len(bin_recs)
            if n_bin > 0:
                w = 1.0 / (1.0 + n_bin / 100.0)
                for tid in bin_recs[:100]:
                    cand_scores[tid] += w

        # Sort by blocker IDF score descending (hardest candidates first)
        sorted_blocker_cands = sorted(cand_scores.items(), key=lambda x: x[1], reverse=True)[:30]
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

    # If shortfall, pad from general blocker collisions
    if len(neg_pairs) < n_neg_target:
        print(f"  Shortfall of {n_neg_target - len(neg_pairs)} negs, drawing additional blocker collisions...")
        for sid, (s_name, s_addr, s_ctry) in s1_dict.items():
            if len(neg_pairs) >= n_neg_target:
                break
            cand_scores = defaultdict(float)
            for k in extract_v3_keys(s_name, s_addr, s_ctry):
                bin_recs = index.get(k, [])
                if bin_recs:
                    for tid in bin_recs[:20]:
                        if tid not in gt_map[sid] and tid in target_dict:
                            neg_pairs.append((sid, tid))
                            if len(neg_pairs) >= n_neg_target:
                                break
                    if len(neg_pairs) >= n_neg_target:
                        break

    print(f"Successfully mined {len(neg_pairs):,} ACTUAL BLOCKER HARD NEGATIVES in {time.time()-t_mine:.2f}s!")
    del index
    gc.collect()

    # 5. Extract 10 RapidFuzz & Structural Features for All 2,000,000 Pairs
    print("\n[STEP 5/5] Extracting 10 high-precision features for 2,000,000 pairs...")
    t_feat = time.time()
    all_pairs = pos_pairs[:n_pos_target] + neg_pairs[:n_neg_target]
    all_labels = np.array([1]*len(pos_pairs[:n_pos_target]) + [0]*len(neg_pairs[:n_neg_target]), dtype=np.uint8)

    # Shuffle
    perm = np.random.RandomState(42).permutation(len(all_pairs))
    shuffled_pairs = [all_pairs[i] for i in perm]
    y_all = all_labels[perm]

    X_mat = np.zeros((len(shuffled_pairs), 10), dtype=np.float32)

    for i, (sid, tid) in enumerate(shuffled_pairs):
        s_name, s_addr, _ = s1_dict[sid]
        t_name, t_addr, _, is_s2 = target_dict[tid]

        # 10 Lexical & Structural Features
        nf = fuzz.ratio(s_name, t_name) / 100.0
        ntset = fuzz.token_set_ratio(s_name, t_name) / 100.0
        ntsort = fuzz.token_sort_ratio(s_name, t_name) / 100.0
        npart = fuzz.partial_ratio(s_name, t_name) / 100.0
        nld = abs(len(s_name) - len(t_name))
        af = fuzz.ratio(s_addr, t_addr) / 100.0
        atset = fuzz.token_set_ratio(s_addr, t_addr) / 100.0

        pins_a = re.findall(r'\b\d{5,6}\b', s_addr)
        pins_b = re.findall(r'\b\d{5,6}\b', t_addr)
        pin_m = 1.0 if (pins_a and pins_b and pins_a[0] == pins_b[0]) else 0.0

        nums_a = extract_numbers(s_addr)
        nums_b = extract_numbers(t_addr)
        num_m = 1.0 if (nums_a and nums_b and (nums_a & nums_b)) else (0.5 if not nums_a or not nums_b else 0.0)

        X_mat[i] = [nf, ntset, ntsort, npart, nld, af, atset, pin_m, num_m, is_s2]

        if (i + 1) % 500000 == 0:
            print(f"  Extracted {i+1:,} / {len(shuffled_pairs):,} pairs ({((i+1)/len(shuffled_pairs))*100:.1f}%)")

    print(f"Feature matrix built in {time.time()-t_feat:.2f}s: X {X_mat.shape} ({X_mat.nbytes/1e6:.1f} MB), y {y_all.shape}")

    # Cache features to disk
    feat_cache_dir = 'output/v3_features'
    os.makedirs(feat_cache_dir, exist_ok=True)
    f_path = os.path.join(feat_cache_dir, 'v3_train_features_2m.npy')
    l_path = os.path.join(feat_cache_dir, 'v3_train_labels_2m.npy')
    np.save(f_path, X_mat)
    np.save(l_path, y_all)
    print(f"Saved cached binary features: {f_path} ({os.path.getsize(f_path)/1e6:.1f} MB)")
    print(f"Saved cached binary labels:   {l_path} ({os.path.getsize(l_path)/1e6:.1f} MB)")

    # 6. Train Regularized 3-Booster Ensemble
    print("\n" + "=" * 75)
    print("TRAINING 3-BOOSTER ENSEMBLE ON ACTUAL BLOCKER HARD NEGATIVES")
    print("=" * 75)
    X_train, X_val, y_train, y_val = train_test_split(X_mat, y_all, test_size=0.15, random_state=42, stratify=y_all)
    print(f"Train split: {len(X_train):,} | Validation split: {len(X_val):,}")

    model_dir = 'V3-Model-Score-0.742/models'
    os.makedirs(model_dir, exist_ok=True)

    # LightGBM
    print("\nTraining LightGBM (max_depth=6, L1=2.0, L2=5.0)...")
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
        'feature_fraction': 0.75,
        'bagging_fraction': 0.75,
        'bagging_freq': 1,
        'verbose': -1,
        'random_state': 42,
        'n_jobs': 4
    }
    t_lgb = time.time()
    lgb_model = lgb.train(lgb_params, lgb_train, num_boost_round=250, valid_sets=[lgb_val], callbacks=[lgb.log_evaluation(50)])
    p_val_lgb = lgb_model.predict(X_val)
    auc_lgb = roc_auc_score(y_val, p_val_lgb)
    print(f"LightGBM trained in {time.time()-t_lgb:.2f}s | Real Blocker Val AUC: {auc_lgb:.5f}")
    lgb_model.save_model(os.path.join(model_dir, 'v3_lightgbm.txt'))

    # XGBoost
    print("\nTraining XGBoost (max_depth=5, L1=2.0, L2=5.0)...")
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    xgb_params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'learning_rate': 0.08,
        'max_depth': 5,
        'alpha': 2.0,
        'lambda': 5.0,
        'subsample': 0.75,
        'colsample_bytree': 0.75,
        'random_state': 42,
        'tree_method': 'hist',
        'n_jobs': 4
    }
    t_xgb = time.time()
    xgb_model = xgb.train(xgb_params, dtrain, num_boost_round=250, evals=[(dval, 'val')], verbose_eval=50)
    p_val_xgb = xgb_model.predict(dval)
    auc_xgb = roc_auc_score(y_val, p_val_xgb)
    print(f"XGBoost trained in {time.time()-t_xgb:.2f}s | Real Blocker Val AUC: {auc_xgb:.5f}")
    xgb_model.save_model(os.path.join(model_dir, 'v3_xgboost.json'))

    # CatBoost
    print("\nTraining CatBoost (depth=6, l2_leaf_reg=5.0)...")
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
    print(f"CatBoost trained in {time.time()-t_cat:.2f}s | Real Blocker Val AUC: {auc_cat:.5f}")
    cat_model.save_model(os.path.join(model_dir, 'v3_catboost.cbm'))

    # Ensemble Blending
    p_val_ens = (0.35 * p_val_lgb) + (0.35 * p_val_xgb) + (0.30 * p_val_cat)
    auc_ens = roc_auc_score(y_val, p_val_ens)
    print(f"\n3-Booster Blended Ensemble Real Blocker Val AUC: {auc_ens:.5f}")

    # Optimal threshold search for Macro F0.5
    best_th = 0.50
    best_f05 = 0.0
    best_p, best_r = 0.0, 0.0
    for th in np.linspace(0.40, 0.85, 46):
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

    # Save ensemble configuration
    config = {
        'models': ['v3_lightgbm.txt', 'v3_xgboost.json', 'v3_catboost.cbm'],
        'weights': {'lightgbm': 0.35, 'xgboost': 0.35, 'catboost': 0.30},
        'reranker': 'BAAI/bge-reranker-v2-m3',
        'optimal_threshold': float(best_th),
        'real_blocker_validation_auc': float(auc_ens),
        'real_blocker_validation_f05': float(best_f05),
        'real_precision': float(best_p),
        'real_recall': float(best_r)
    }
    cfg_path = os.path.join(model_dir, 'v3_ensemble_config.json')
    with open(cfg_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Saved updated ensemble configuration to {cfg_path}")

    total_time = time.time() - t0
    print("\n" + "=" * 75)
    print(f"PIPELINE COMPLETE IN {total_time:.2f}s ({total_time/60:.1f} minutes)")
    print("=" * 75)

if __name__ == '__main__':
    main()
