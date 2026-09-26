# V2 Model Pipeline — Macro $F_{0.5}$ Score: 0.418

## Overview
This folder contains the complete runnable artifacts, scripts, models, and results for the **V2 High-Recall Transliterated Candidate Generator & GBDT Ensemble**, which increased the competition leaderboard score from **0.169 $\rightarrow$ 0.418** (+0.249 increase).

---

## Key Breakthroughs Over Baseline (0.169)
1. **Transliterated Multi-Key Blocking:**
   - Applied `unidecode` phonetic transliteration across all Indic scripts (Devanagari/Hindi, Tamil) and French accented characters.
   - Ground truth match recall inside candidates jumped from **8.37% $\rightarrow$ 89.21%**.
   - Capped candidate pool at **25 candidates per S1 entity** (total 43.3M candidate pairs).
2. **Locality + Street Number Composite Hashing:**
   - Composite keys `nl_{number}_{locality}` and 5/6-digit PIN/ZIP code matching captured true duplicates with zero name overlap (e.g. DBA trade aliases).
3. **Elimination of the Singleton Penalty:**
   - Shifted from static high threshold ($\tau=0.78$ which predicted 68.4% empty lists) to **Entity-Level Top-1 Selection**.
   - Percentage of S1 entities matched increased from **31.6% $\rightarrow$ 98.2%** (1,700,757 entities).

---

## Architecture & Model
- **Training Set:** 1,200,000 balanced pairs (600,000 ground-truth true positives + 600,000 locality-mined hard negatives).
- **Features (10):**
  1. `name_ratio` (rapidfuzz ratio on unidecode names)
  2. `name_token_sort` (token sort ratio)
  3. `name_token_set` (token set ratio)
  4. `name_len_diff` (absolute character length difference)
  5. `addr_ratio` (rapidfuzz ratio on unidecode addresses)
  6. `addr_token_set` (address token set ratio)
  7. `num_match` (street number match flag: 1.0, 0.5, 0.0)
  8. `pin_match` (PIN / ZIP code match flag)
  9. `is_s2` (1.0 for S2 targets, 0.0 for S3 targets)
  10. `sem_proxy` (multi-field phonetic & locality overlap proxy)
- **Classifier:** 50-50 Ensemble of LightGBM (180 trees) + XGBoost (180 trees, `tree_method='hist'`).
- **Validation Metric:** 180,000 Validation Pairs ROC-AUC: **`0.9994`**.

---

## Directory Structure
- `scripts/generate_test_candidates_v2.py`: Generates the 43.3M candidate pool.
- `scripts/train_multilingual_pipeline.py`: Assembles 1.2M training set, extracts 10 features, trains ensemble, and generates test predictions.
- `models/v2_lightgbm.txt`: Trained LightGBM booster.
- `models/v2_xgboost.json`: Trained XGBoost booster.
- `models/v2_features_1m.npy`: Cached 1.2M feature matrix (48 MB).
- `models/v2_labels_1m.npy`: Cached 1.2M labels array.
- `results/matching_results_score_0.418.tsv`: Official submission file that scored 0.418.
