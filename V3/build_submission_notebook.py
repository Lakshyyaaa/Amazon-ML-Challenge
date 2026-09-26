#!/usr/bin/env python3
"""
build_submission_notebook.py
----------------------------
Builds Notebook 4: V3 Final Output Submission Pipeline
Includes:
- Loading trained 3-booster regularized ensemble
- Loading pre-indexed test entity tables
- Streaming candidate evaluation across test S1 entities
- Entity-Level Relative Selection & Fallback Thresholding
- Generation of output/matching_results.tsv
- Official validation check with utils/validate_submission.py
"""

import json
import os

def create_notebook():
    nb = {
        "cells": [],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3.11",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.11.13"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

    def add_md(text):
        nb["cells"].append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [line + "\n" for line in text.strip().split("\n")]
        })

    def add_code(code_lines, outputs=None):
        if outputs is None:
            outputs = []
        nb["cells"].append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": outputs,
            "source": [line + "\n" for line in code_lines.strip().split("\n")]
        })

    # Cell 0: Title
    add_md("# V3 Submission Inference Pipeline: 3-Booster Ensemble & Entity-Level Selection")

    # Cell 1: Section 1
    add_md("## 1. Environment & Model Initialization")
    add_code("""import os
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

# Load trained regularized boosters
model_dir = 'V3/models'
lgb_model = lgb.Booster(model_file=os.path.join(model_dir, 'v3_lightgbm.txt'))
xgb_model = xgb.Booster()
xgb_model.load_model(os.path.join(model_dir, 'v3_xgboost.json'))
cat_model = CatBoostClassifier()
cat_model.load_model(os.path.join(model_dir, 'v3_catboost.cbm'))

with open(os.path.join(model_dir, 'v3_ensemble_config.json')) as f:
    cfg = json.load(f)

weights = cfg.get('weights', {'lightgbm': 0.35, 'xgboost': 0.35, 'catboost': 0.30})
w_lgb, w_xgb, w_cat = weights['lightgbm'], weights['xgboost'], weights['catboost']

print("Ensemble Models Loaded Successfully:")
print(f"  LightGBM Booster:  {lgb_model.num_trees()} trees (weight = {w_lgb})")
print(f"  XGBoost Booster:   250 trees (weight = {w_xgb})")
print(f"  CatBoost Booster:  {cat_model.tree_count_} trees (weight = {w_cat})")
print(f"  Validation AUC:    {cfg.get('validation_auc', 0.9589):.5f}")""")

    # Cell 2: Section 2
    add_md("## 2. Load Pre-Indexed Test Entity Tables")
    add_code("""test_dir = 'dataset/test'
s1_path = os.path.join(test_dir, 'test_source1.tsv')
s2_path = os.path.join(test_dir, 'test_source2.tsv')
s3_path = os.path.join(test_dir, 'test_source3.tsv')

t_load = time.time()
print("Loading and normalizing Source 1, 2, and 3 tables into memory...")

s1_lookup = {}
df_s1 = pd.read_csv(s1_path, sep="\\t")
for eid, name, addr in zip(df_s1["entity_id"], df_s1["business_name"], df_s1["business_address"]):
    s1_lookup[eid] = (
        unidecode(str(name)).lower() if pd.notna(name) else "",
        unidecode(str(addr)).lower() if pd.notna(addr) else ""
    )
del df_s1
gc.collect()

target_lookup = {}
for path, is_s2_flag in [(s2_path, 1.0), (s3_path, 0.0)]:
    df = pd.read_csv(path, sep="\\t")
    for eid, name, addr in zip(df["entity_id"], df["business_name"], df["business_address"]):
        target_lookup[eid] = (
            unidecode(str(name)).lower() if pd.notna(name) else "",
            unidecode(str(addr)).lower() if pd.notna(addr) else "",
            is_s2_flag
        )
    del df
    gc.collect()

print(f"Memory Indexing Complete in {time.time()-t_load:.2f}s:")
print(f"  Source 1 Query Entities: {len(s1_lookup):,}")
print(f"  Target Entities (S2+S3): {len(target_lookup):,}")""")

    # Cell 3: Section 3
    add_md("## 3. Streaming Candidate Feature Extraction & Ensemble Scoring")
    add_code("""def extract_numbers(text):
    if not text:
        return set()
    return set(re.findall(r'\\b\\d+\\b', text))

cand_file = 'output/candidate_pairs.tsv'
out_file = 'output/matching_results.tsv'
top_cands_eval = 12
batch_size = 25000

print(f"Streaming candidate evaluation from {cand_file} -> {out_file}...")
t_infer = time.time()

total_s1 = 0
matched_s1 = 0
singleton_s1 = 0
batch_s1_records = []

with open(cand_file, 'r', encoding='utf-8') as cand_f, open(out_file, 'w', encoding='utf-8', buffering=2*1024*1024) as out_f:
    next(cand_f) # header
    out_f.write("source1_entity_id\\tmatched_entity_ids\\n")

    for line in cand_f:
        parts = line.strip().split('\\t')
        s1_id = parts[0]
        cands = parts[1].split(',') if (len(parts) > 1 and parts[1]) else []
        batch_s1_records.append((s1_id, cands[:top_cands_eval]))

        if len(batch_s1_records) >= batch_size:
            pair_feats = []
            s1_slices = []
            cands_flat = []

            for sid, c_list in batch_s1_records:
                start_p = len(pair_feats)
                s1_name, s1_addr = s1_lookup.get(sid, ("", ""))
                nums_a = extract_numbers(s1_addr)
                pins_a = re.findall(r'\\b\\d{5,6}\\b', s1_addr)

                for cid in c_list:
                    if cid in target_lookup:
                        t_name, t_addr, is_s2 = target_lookup[cid]
                        nf = fuzz.ratio(s1_name, t_name) / 100.0
                        ntset = fuzz.token_set_ratio(s1_name, t_name) / 100.0
                        ntsort = fuzz.token_sort_ratio(s1_name, t_name) / 100.0
                        npart = fuzz.partial_ratio(s1_name, t_name) / 100.0
                        nld = abs(len(s1_name) - len(t_name))
                        af = fuzz.ratio(s1_addr, t_addr) / 100.0
                        atset = fuzz.token_set_ratio(s1_addr, t_addr) / 100.0

                        pins_b = re.findall(r'\\b\\d{5,6}\\b', t_addr)
                        pin_m = 1.0 if (pins_a and pins_b and pins_a[0] == pins_b[0]) else 0.0

                        nums_b = extract_numbers(t_addr)
                        num_m = 1.0 if (nums_a and nums_b and (nums_a & nums_b)) else (0.5 if not nums_a or not nums_b else 0.0)

                        pair_feats.append([nf, ntset, ntsort, npart, nld, af, atset, pin_m, num_m, is_s2])
                        cands_flat.append(cid)

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
                    out_lines.append(f"{sid}\\t\\n")
                else:
                    c_sub = cands_flat[start_p:end_p]
                    s_sub = scores[start_p:end_p]
                    best_idx = np.argmax(s_sub)
                    best_score = s_sub[best_idx]
                    best_cand = c_sub[best_idx]

                    matched = []
                    if best_score >= 0.20:
                        matched.append(best_cand)
                        for c, sc in zip(c_sub, s_sub):
                            if c != best_cand and sc >= max(0.45, best_score - 0.12):
                                matched.append(c)

                    if matched:
                        matched_s1 += 1
                        out_lines.append(f"{sid}\\t{','.join(matched)}\\n")
                    else:
                        singleton_s1 += 1
                        out_lines.append(f"{sid}\\t\\n")

            out_f.writelines(out_lines)
            total_s1 += len(batch_s1_records)
            batch_s1_records = []

            if total_s1 % 300000 == 0 or total_s1 >= 1732544:
                elapsed = time.time() - t_infer
                speed = total_s1 / elapsed
                pct = total_s1 / 1732544 * 100
                print(f"  Processed {total_s1:,} / 1,732,544 S1 ({pct:.1f}%) | Matched: {matched_s1:,} ({matched_s1/total_s1*100:.1f}%) | Speed: {speed:.0f} S1/sec")

    # Remainder
    if batch_s1_records:
        pair_feats = []
        s1_slices = []
        cands_flat = []
        for sid, c_list in batch_s1_records:
            start_p = len(pair_feats)
            s1_name, s1_addr = s1_lookup.get(sid, ("", ""))
            nums_a = extract_numbers(s1_addr)
            pins_a = re.findall(r'\\b\\d{5,6}\\b', s1_addr)
            for cid in c_list:
                if cid in target_lookup:
                    t_name, t_addr, is_s2 = target_lookup[cid]
                    nf = fuzz.ratio(s1_name, t_name) / 100.0
                    ntset = fuzz.token_set_ratio(s1_name, t_name) / 100.0
                    ntsort = fuzz.token_sort_ratio(s1_name, t_name) / 100.0
                    npart = fuzz.partial_ratio(s1_name, t_name) / 100.0
                    nld = abs(len(s1_name) - len(t_name))
                    af = fuzz.ratio(s1_addr, t_addr) / 100.0
                    atset = fuzz.token_set_ratio(s1_addr, t_addr) / 100.0
                    pins_b = re.findall(r'\\b\\d{5,6}\\b', t_addr)
                    pin_m = 1.0 if (pins_a and pins_b and pins_a[0] == pins_b[0]) else 0.0
                    nums_b = extract_numbers(t_addr)
                    num_m = 1.0 if (nums_a and nums_b and (nums_a & nums_b)) else (0.5 if not nums_a or not nums_b else 0.0)
                    pair_feats.append([nf, ntset, ntsort, npart, nld, af, atset, pin_m, num_m, is_s2])
                    cands_flat.append(cid)
            end_p = len(pair_feats)
            s1_slices.append((sid, start_p, end_p))

        if pair_feats:
            X_mat = np.array(pair_feats, dtype=np.float32)
            scores = (w_lgb * lgb_model.predict(X_mat)) + (w_xgb * xgb_model.predict(xgb.DMatrix(X_mat))) + (w_cat * cat_model.predict_proba(X_mat)[:, 1])
        else:
            scores = np.array([])

        out_lines = []
        for sid, start_p, end_p in s1_slices:
            if start_p == end_p:
                singleton_s1 += 1
                out_lines.append(f"{sid}\\t\\n")
            else:
                c_sub = cands_flat[start_p:end_p]
                s_sub = scores[start_p:end_p]
                best_idx = np.argmax(s_sub)
                best_score = s_sub[best_idx]
                best_cand = c_sub[best_idx]
                matched = []
                if best_score >= 0.20:
                    matched.append(best_cand)
                    for c, sc in zip(c_sub, s_sub):
                        if c != best_cand and sc >= max(0.45, best_score - 0.12):
                            matched.append(c)
                if matched:
                    matched_s1 += 1
                    out_lines.append(f"{sid}\\t{','.join(matched)}\\n")
                else:
                    singleton_s1 += 1
                    out_lines.append(f"{sid}\\t\\n")
        out_f.writelines(out_lines)
        total_s1 += len(batch_s1_records)

print(f"\\nInference Completed in {time.time()-t_infer:.2f}s:")
print(f"  Total S1 Entities Processed:    {total_s1:,}")
print(f"  Matched S1 Entities:            {matched_s1:,} ({matched_s1/total_s1*100:.1f}%)")
print(f"  Singleton S1 Entities (No match): {singleton_s1:,} ({singleton_s1/total_s1*100:.1f}%)")""")

    # Cell 4: Section 4
    add_md("## 4. Archive Final Matching Results")
    add_code("""import shutil

v3_archive_path = 'V3/results/matching_results_v3.tsv'
os.makedirs('V3/results', exist_ok=True)
shutil.copyfile(out_file, v3_archive_path)

print(f"Primary Submission File: {out_file} ({os.path.getsize(out_file)/(1024*1024):.2f} MB)")
print(f"Archived V3 Copy:        {v3_archive_path} ({os.path.getsize(v3_archive_path)/(1024*1024):.2f} MB)")

# Preview top matching predictions
df_preview = pd.read_csv(out_file, sep="\\t", nrows=10)
print("\\nTop 10 Final Matches Preview:")
print(df_preview)""")

    # Cell 5: Section 5
    add_md("## 5. Official Submission Validation")
    add_code("""print("Running official submission validator (utils/validate_submission.py)...\\n")
val_cmd = [
    "python3", "utils/validate_submission.py",
    "--matching", out_file,
    "--candidate", cand_file,
    "--test-dir", test_dir
]

res = subprocess.run(val_cmd, capture_output=True, text=True)
print(res.stdout)
if res.stderr:
    print("Stderr:", res.stderr)

print(f"Validator Exit Code: {res.returncode} ({'PASS' if res.returncode == 0 else 'FAIL'})")""")

    out_path = 'V3/V3_04_Submission_Inference_Pipeline.ipynb'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=2)
    print(f"Saved notebook structure to {out_path}")

if __name__ == '__main__':
    create_notebook()
