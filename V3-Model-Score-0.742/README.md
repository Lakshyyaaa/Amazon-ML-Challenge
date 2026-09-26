# V3 Model Pipeline — Macro $F_{0.5}$ Score: 0.742

## Overview
This folder contains the complete reproducible notebooks, models, scripts, and results for the **V3 Multi-Route High-Recall Candidate Generator & Blocker-Negative Trained 3-Booster Ensemble**, which advanced the competition leaderboard score from **0.418 $\rightarrow$ 0.601 $\rightarrow$ 0.742** (+0.324 increase over V2, and over 4.3× higher than the 0.169 baseline).

---

## Performance Progression

| Metric / Stage | V1 (Baseline) | V2 (Multilingual) | V3 (Initial 12-Cand) | **V3 (Full 40-Cand Blocker-Negative Ensemble)** |
| :--- | :--- | :--- | :--- | :--- |
| **Blocking Ground-Truth Recall** | 8.37% | 71.49% | 85.35% (top 40 pool) | **85.35%** (fully scored across all 40 cands) |
| **S1 Entity Candidate Coverage** | 0.29% | 89.21% | 95.82% | **95.82%** (1,660,113 entities) |
| **Candidate Evaluation Depth** | Top 12 | Top 25 | Top 12 (truncated) | **Top 40 (Full Candidate Pool Scored)** |
| **Training Hard Negatives** | Random collisions | Shared country/name | Random bucket | **Actual Top Blocker Non-Match Collisions** |
| **Matches Predicted per S1** | ~1.00 | ~1.00 | 5.07 *(+47% over-prediction)* | **3.30** *(matches real Ground Truth mean: 3.45)* |
| **Singleton Handling** | 68.4% empty | Forced match | 1.0% empty | **2.7% preserved empty (46,441 entities)** |
| **Booster Models Trained** | Rule-based GBDT | LGBM + XGBoost | LGBM + XGB + Cat | **LightGBM + XGBoost + CatBoost** |
| **Leaderboard Macro $F_{0.5}$** | **0.169** | **0.418** | **0.601** | **`0.742`** *(+0.141 leap)* |

---

## Detailed List of Changes Made in V3 (0.742)

1. **Full 40-Candidate Evaluation (Eliminating Truncation)**:
   - Previous inference runs artificially capped candidate scoring at `top_cands_eval = 12`, discarding candidates 13 through 40 where ~23% of true matches reside.
   - Upgraded inference to evaluate **all 40 candidates** in `candidate_pairs.tsv` with early heuristic pre-filtering, unlocking the complete **85.35% blocker recall**.

2. **Mining Actual Blocker Hard Negatives (Fixing Negative Bias)**:
   - Replaced generic bucket negatives with **1,000,000 actual blocker hard negatives** mined by querying the training S1 queries against the 6-route inverted index blocker.
   - The negatives are the **top IDF-scoring non-ground-truth candidates returned by the blocker**, forcing the trees to learn the exact subtle boundaries between true matches and hard acoustic/token distractors.
   - Trained LightGBM (Val AUC: 0.9970), XGBoost (Val AUC: 0.9968), and CatBoost (Val AUC: 0.9967) with an ensemble blended Val AUC of **0.99692**.

3. **Eliminating False-Positive Chain Branch Over-Selection**:
   - In ground truth, an S1 entity matches an average of **3.45 targets** (1.77 in S2, 1.89 in S3).
   - The previous unconstrained inference predicted **5.07 matches per S1** (+47% false positive flood of duplicate branches), which dragged entity precision down to ~58% and capped $F_{0.5}$ at 0.600.
   - Enforced maximum 3 matches from S2 and maximum 3 from S3 with calibrated relative thresholding (`sc >= max(0.55, best_score - 0.08)`).
   - Matches per entity dropped from **5.07 $\rightarrow$ 3.30**, restoring high precision and spiking the $F_{0.5}$ score.

4. **Calibrated Confidence Cutoff & Singleton Preservation**:
   - Implemented `best_score < 0.50` thresholding: if no candidate achieves confidence, the entity is output as empty.
   - Preserved **46,441 entities (2.7%) as clean singletons**, preventing fatal precision losses on unmatchable entities.

5. **Reproducibility & Notebook Integrity**:
   - Added [build_real_training_data_v3.py](file:///Users/lakshyasantani/Desktop/Amazon%20ML%20Challenge/build_real_training_data_v3.py) to reproducibly assemble the real training pairs directly from `dataset/train/`.
   - Completely deleted the legacy `np.random.beta` cell from `V3_03_Model_Training_And_Reranking.ipynb` and restored proper executable imports and loader cells.

---

## Directory Structure & Artifacts

- **Notebooks**:
  - `V3_01_Candidate_Pairs_Generation.ipynb`: 6-route blocking generator creating the 69.3M candidate pairs pool.
  - `V3_02_Feature_Extraction.ipynb`: RapidFuzz lexical extraction & 1024-d `multilingual-e5-large-instruct` cosine similarity matrix.
  - `V3_03_Model_Training_And_Reranking.ipynb`: 2M genuine blocker-negative real-data training, 3 regularized boosting models, and BGE reranker fusion.
  - `V3_04_Submission_Inference_Pipeline.ipynb`: Streaming test inference across all 1,732,544 entities, 40-candidate scoring, relative selection, and submission validation.
- **Features (`features/`)**:
  - `v3_features_sample.parquet`: Annotated DataFrame with lexical and semantic cosine features.
  - `v3_feature_matrix_sample.npy`: Assembled numerical feature matrix.
- **Trained Models (`models/`)**:
  - `v3_lightgbm.txt`: Regularized LightGBM booster.
  - `v3_xgboost.json`: Regularized XGBoost booster.
  - `v3_catboost.cbm`: Regularized CatBoost booster.
  - `v3_ensemble_config.json`: Ensemble weights and decision parameters (tau = 0.770).
- **Results (`results/`)**:
  - `matching_results_v3.tsv`: Validated submission file that scored **`0.742`** on the official leaderboard.
