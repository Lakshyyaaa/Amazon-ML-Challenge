# V3 Model Pipeline — Macro $F_{0.5}$ Score: 0.601

## Overview
This folder contains the complete reproducible notebooks, models, scripts, and results for the **V3 Multi-Route High-Recall Candidate Generator & Regularized 3-Booster Ensemble**, which advanced the competition leaderboard score from **0.418 $\rightarrow$ 0.601** (+0.183 increase, and over 3.5× higher than the 0.169 baseline).

---

## Performance Progression

| Metric / Stage | V1 (Baseline) | V2 (Multilingual) | **V3 (Multi-Route Regularized Ensemble)** |
| :--- | :--- | :--- | :--- |
| **Blocking Ground-Truth Recall** | 8.37% | 71.49% | **85.35%** (+96,758 matches recovered) |
| **S1 Entity Candidate Coverage** | 0.29% | 89.21% | **95.82%** (1,660,113 entities) |
| **Candidate Depth ($K$)** | Top 12 | Top 25 | **Top 40 (69,301,760 total pairs)** |
| **Booster Models Trained** | Rule-based GBDT | LGBM + XGBoost | **LightGBM + XGBoost + CatBoost** |
| **Cross-Encoder Re-Ranking** | None | None | **BAAI/bge-reranker-v2-m3** |
| **Anti-Overfitting Defenses** | None | Mild | **Tree Depth (5/6), $L_1/L_2$ Penalties, Realistic Boundary Overlap** |
| **Leaderboard Macro $F_{0.5}$** | **0.169** | **0.418** | **`0.601`** *(+0.183 leap)* |

---

## Key Breakthroughs Over V2 (0.418)

1. **6-Route Inverted Index Blocking**:
   - Phonetic acoustic keys (`Soundex`, `Double Metaphone` via `jellyfish`).
   - Character 4-Grams on compressed alphanumeric strings.
   - Word Bi-Grams for word-order tolerance in company names.
   - Address composite keys: Building numbers + Locality hashes + 5/6-digit PIN/ZIP codes.
   - De-prefixing of digital identifiers (`@`, `www.`, `.com`).
   - Dynamic sub-linear IDF damping: $\text{weight} = \frac{1.0}{1.0 + N_{\text{bin}} / 100.0}$.
   - True match retrieval inside candidates surged to **85.35%**.

2. **3-Model Boosting Diversity (LightGBM + XGBoost + CatBoost)**:
   - Combines histogram leaf-wise splitting (LightGBM), depth-wise gradient boosting (XGBoost), and symmetric oblivious decision trees (CatBoost).
   - Soft-voting blend: $0.35 \times \text{LGB} + 0.35 \times \text{XGB} + 0.30 \times \text{Cat}$.

3. **100% Genuine Real-Data Training & Anti-Overfitting Defenses**:
   - 1,000,000 ground-truth positive pairs mined directly from `dataset/train/train_ground_truth.tsv` across genuine S1-S2 and S1-S3 matches.
   - 1,000,000 real hard-negative pairs mined directly from genuine blocking collisions across `train_source1/2/3.tsv` (same country and blocking bin, but non-matching). Zero synthetic data.
   - Restricted tree depth (`max_depth=5/6`), applied strong regularization ($L_1=2.0$, $L_2=5.0$), and subsampled features ($0.75$).
   - Train-validation generalization gap was held to **$\approx 0.00004$** (Train AUC: 0.99956, Real Val AUC: 0.99952).

4. **Neural Cross-Encoder Re-Ranking (`BAAI/bge-reranker-v2-m3`)**:
   - Evaluates full concatenated text pairs using deep cross-attention to disambiguate borderline pairs with tied lexical scores.

---

## Directory Structure & Artifacts

- **Notebooks**:
  - `V3_01_Candidate_Pairs_Generation.ipynb`: 6-route blocking generator creating the 69.3M candidate pairs pool.
  - `V3_02_Feature_Extraction.ipynb`: RapidFuzz lexical extraction & 1024-d `multilingual-e5-large-instruct` cosine similarity matrix.
  - `V3_03_Model_Training_And_Reranking.ipynb`: 2M genuine balanced real-data training, 3 regularized boosting models, and BGE reranker fusion.
  - `V3_04_Submission_Inference_Pipeline.ipynb`: Streaming test inference across all 1,732,544 entities, relative selection, and submission validation.
- **Features (`features/`)**:
  - `v3_features_sample.parquet`: Annotated DataFrame with lexical and semantic cosine features.
  - `v3_feature_matrix_sample.npy`: Assembled numerical feature matrix.
- **Trained Models (`models/`)**:
  - `v3_lightgbm.txt`: Regularized LightGBM booster.
  - `v3_xgboost.json`: Regularized XGBoost booster.
  - `v3_catboost.cbm`: Regularized CatBoost booster.
  - `v3_ensemble_config.json`: Ensemble weights and decision parameters.
- **Results (`results/`)**:
  - `matching_results_v3.tsv`: Validated submission file that scored **`0.601`** on the official leaderboard.
