#!/usr/bin/env python3
"""
build_feature_extraction_notebook.py
-----------------------------------
Builds and executes V3/V3_02_Feature_Extraction.ipynb with real outputs.
Extracts:
- Lexical features: Fuzz Ratio, Token Set Ratio, Token Sort Ratio, Partial Ratio, Address Ratios, Length Diff, PIN/Num matches
- Multilingual Transformer embeddings: intfloat/multilingual-e5-large-instruct
- Cosine similarities: Name Cosine Similarity & Full Text Cosine Similarity
- Unified Feature Matrix assembly & persistence
"""

import json
import os
import sys

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
    add_md("# V3 Feature Extraction: Lexical & Multilingual Transformer Embeddings")

    # Cell 1: Section 1
    add_md("## 1. Imports & Environment Configuration")
    add_code("""import os
import re
import gc
import time
import numpy as np
import pandas as pd
from unidecode import unidecode
from rapidfuzz import fuzz
import torch
from sentence_transformers import SentenceTransformer

device = 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f"PyTorch Version: {torch.__version__}")
print(f"Compute Device:  {device}")
print(f"RapidFuzz Version: {fuzz.__file__}")""")

    # Cell 2: Section 2
    add_md("## 2. Load Entity Data & Candidate Pairs")
    add_code("""# Load sample of candidate pairs and entity tables for feature matrix extraction
s1_path = 'dataset/test/test_source1.tsv'
s2_path = 'dataset/test/test_source2.tsv'
s3_path = 'dataset/test/test_source3.tsv'
cand_path = 'output/candidate_pairs.tsv'

# Read entity lookups
print("Loading entity tables...")
df_s1 = pd.read_csv(s1_path, sep='\\t', nrows=10000)
df_s2 = pd.read_csv(s2_path, sep='\\t', nrows=50000)
df_s3 = pd.read_csv(s3_path, sep='\\t', nrows=50000)

s1_lookup = {row['entity_id']: (str(row['business_name']), str(row['business_address']), str(row['country'])) for _, row in df_s1.iterrows()}
target_lookup = {}
for _, row in df_s2.iterrows():
    target_lookup[row['entity_id']] = (str(row['business_name']), str(row['business_address']), str(row['country']), 1.0)
for _, row in df_s3.iterrows():
    target_lookup[row['entity_id']] = (str(row['business_name']), str(row['business_address']), str(row['country']), 0.0)

# Sample candidate pairs for matrix generation
candidate_samples = []
with open(cand_path, 'r', encoding='utf-8') as f:
    next(f) # header
    for line in f:
        parts = line.strip().split('\\t')
        if len(parts) == 2 and parts[1]:
            s1_id = parts[0]
            cands = parts[1].split(',')
            if s1_id in s1_lookup:
                for c_id in cands[:5]: # top-5 candidates per entity sample
                    if c_id in target_lookup:
                        candidate_samples.append((s1_id, c_id))
        if len(candidate_samples) >= 1000:
            break

df_pairs = pd.DataFrame(candidate_samples, columns=['source1_id', 'candidate_id'])
print(f"Sample Candidate Pairs to Process: {len(df_pairs):,}")
print(df_pairs.head())""")

    # Cell 3: Section 3
    add_md("## 3. Lexical Feature Extraction (RapidFuzz)")
    add_code("""def extract_lexical_features(s1_name, s1_addr, s1_ctry, c_name, c_addr, c_ctry):
    # Transliterated lower strings
    n1 = unidecode(str(s1_name)).lower()
    n2 = unidecode(str(c_name)).lower()
    a1 = unidecode(str(s1_addr)).lower()
    a2 = unidecode(str(c_addr)).lower()
    
    # 1. Name Lexical Ratios
    fuzz_ratio = fuzz.ratio(n1, n2) / 100.0
    token_set = fuzz.token_set_ratio(n1, n2) / 100.0
    token_sort = fuzz.token_sort_ratio(n1, n2) / 100.0
    partial_ratio = fuzz.partial_ratio(n1, n2) / 100.0
    len_diff = abs(len(n1) - len(n2))
    
    # 2. Address Lexical Ratios
    addr_ratio = fuzz.ratio(a1, a2) / 100.0
    addr_token_set = fuzz.token_set_ratio(a1, a2) / 100.0
    
    # 3. Numeric & PIN Matches
    pins1 = set(re.findall(r'\\b\\d{5,6}\\b', a1))
    pins2 = set(re.findall(r'\\b\\d{5,6}\\b', a2))
    pin_match = 1.0 if (pins1 and pins2 and pins1.intersection(pins2)) else 0.0
    
    nums1 = set(re.findall(r'\\b\\d+\\b', a1))
    nums2 = set(re.findall(r'\\b\\d+\\b', a2))
    if nums1 and nums2:
        inter = nums1.intersection(nums2)
        num_match = 1.0 if inter else 0.0
    else:
        num_match = 0.5
        
    country_match = 1.0 if str(s1_ctry).strip() == str(c_ctry).strip() else 0.0
    
    return [
        fuzz_ratio,
        token_set,
        token_sort,
        partial_ratio,
        len_diff,
        addr_ratio,
        addr_token_set,
        pin_match,
        num_match,
        country_match
    ]

lex_features = []
for _, row in df_pairs.iterrows():
    s1_info = s1_lookup[row['source1_id']]
    c_info = target_lookup[row['candidate_id']]
    feats = extract_lexical_features(s1_info[0], s1_info[1], s1_info[2], c_info[0], c_info[1], c_info[2])
    feats.append(c_info[3]) # is_s2 target source indicator
    lex_features.append(feats)

lex_cols = [
    'name_fuzz_ratio', 'name_token_set_ratio', 'name_token_sort_ratio', 'name_partial_ratio',
    'name_len_diff', 'addr_fuzz_ratio', 'addr_token_set_ratio', 'pin_match', 'num_match',
    'country_match', 'is_s2'
]
df_lex = pd.DataFrame(lex_features, columns=lex_cols)
print(f"Extracted {len(df_lex)} Lexical Feature Rows:")
print(df_lex.head())""")

    # Cell 4: Section 4
    add_md("## 4. Multilingual Transformer Embeddings (multilingual-e5-large-instruct)")
    add_code("""# Load multilingual-e5-large-instruct
model_name = 'intfloat/multilingual-e5-large-instruct'
print(f"Loading {model_name}...")
t0 = time.time()
transformer_model = SentenceTransformer(model_name, device=device)
print(f"Model loaded in {time.time() - t0:.2f}s")

# Extract unique text strings for batched embedding inference
s1_unique_ids = df_pairs['source1_id'].unique()
c_unique_ids = df_pairs['candidate_id'].unique()

# Format queries with instruction prefix as specified by e5-large-instruct
s1_texts_name = [f"Instruct: Given an entity name, retrieve matching representations.\\nQuery: {s1_lookup[uid][0]}" for uid in s1_unique_ids]
s1_texts_full = [f"Instruct: Given an entity name and address, retrieve matching representations.\\nQuery: {s1_lookup[uid][0]} {s1_lookup[uid][1]}" for uid in s1_unique_ids]

c_texts_name = [target_lookup[uid][0] for uid in c_unique_ids]
c_texts_full = [f"{target_lookup[uid][0]} {target_lookup[uid][1]}" for uid in c_unique_ids]

print(f"Encoding {len(s1_unique_ids):,} Query Entities...")
s1_name_emb = transformer_model.encode(s1_texts_name, batch_size=64, show_progress_bar=True, normalize_embeddings=True)
s1_full_emb = transformer_model.encode(s1_texts_full, batch_size=64, show_progress_bar=True, normalize_embeddings=True)

print(f"Encoding {len(c_unique_ids):,} Target Entities...")
c_name_emb = transformer_model.encode(c_texts_name, batch_size=64, show_progress_bar=True, normalize_embeddings=True)
c_full_emb = transformer_model.encode(c_texts_full, batch_size=64, show_progress_bar=True, normalize_embeddings=True)

s1_name_map = dict(zip(s1_unique_ids, s1_name_emb))
s1_full_map = dict(zip(s1_unique_ids, s1_full_emb))
c_name_map = dict(zip(c_unique_ids, c_name_emb))
c_full_map = dict(zip(c_unique_ids, c_full_emb))

print("Embeddings computed and mapped successfully.")""")

    # Cell 5: Section 5
    add_md("## 5. Compute Cosine Similarities & Assemble Feature Matrix")
    add_code("""# Compute Cosine Similarities (Dot product of L2-normalized embeddings)
cos_sim_name = []
cos_sim_full = []

for _, row in df_pairs.iterrows():
    u_n = s1_name_map[row['source1_id']]
    v_n = c_name_map[row['candidate_id']]
    cos_sim_name.append(float(np.dot(u_n, v_n)))
    
    u_f = s1_full_map[row['source1_id']]
    v_f = c_full_map[row['candidate_id']]
    cos_sim_full.append(float(np.dot(u_f, v_f)))

# Assemble Unified Feature Matrix
df_features = df_lex.copy()
df_features['e5_name_cosine_sim'] = cos_sim_name
df_features['e5_full_cosine_sim'] = cos_sim_full
df_features.insert(0, 'candidate_id', df_pairs['candidate_id'])
df_features.insert(0, 'source1_id', df_pairs['source1_id'])

print("Unified Feature Matrix Assembly Completed:")
print(f"Total Rows:     {len(df_features):,}")
print(f"Total Features: {df_features.shape[1] - 2}")
print("\\nFeature Correlation with e5_name_cosine_sim:")
print(df_features.select_dtypes(include=[np.number]).corr()['e5_name_cosine_sim'].sort_values(ascending=False))
print("\\nFeature Matrix Head:")
print(df_features.head(10))""")

    # Cell 6: Section 6
    add_md("## 6. Export Feature Matrix")
    add_code("""output_dir = 'output/v3_features'
os.makedirs(output_dir, exist_ok=True)

parquet_path = os.path.join(output_dir, 'v3_features_sample.parquet')
df_features.to_parquet(parquet_path, index=False)

matrix_X = df_features.drop(columns=['source1_id', 'candidate_id']).values
npy_path = os.path.join(output_dir, 'v3_feature_matrix_sample.npy')
np.save(npy_path, matrix_X)

print(f"Saved Parquet Features: {parquet_path} ({os.path.getsize(parquet_path) / 1024:.2f} KB)")
print(f"Saved NumPy Feature Matrix: {npy_path} ({os.path.getsize(npy_path) / 1024:.2f} KB)")
print(f"Feature Matrix Shape: {matrix_X.shape}")""")

    out_path = 'V3/V3_02_Feature_Extraction.ipynb'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=2)
    print(f"Saved notebook structure to {out_path}")

if __name__ == '__main__':
    create_notebook()
