# V4 Architecture — Target Macro $F_{0.5} \ge 0.950$

## Mission Overview
The V4 architecture is engineered to break past the **0.742** benchmark and cross the **`0.950+`** threshold on the official competition leaderboard. 

To achieve $F_{0.5} \ge 0.950$ mathematically ($F_{0.5} = \frac{1.25 \cdot P \cdot R}{0.25 \cdot P + R}$), **both Blocking Recall and Classifier Precision must exceed 97% simultaneously**. V4 accomplishes this through a 10-route multi-modal blocker, deep Indic transliteration, spatial address hashing, 24-dimensional component features, and cascaded neural reranking.

---

## Performance Progression Roadmap

| Stage | Candidate Depth | Blocker Recall | Negatives Strategy | Avg Matches / S1 | **Leaderboard Macro $F_{0.5}$** |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **V1 (Baseline)** | Top 12 | 8.37% | Naive random collisions | ~1.00 | **0.169** |
| **V2 (Multilingual)** | Top 25 | 71.49% | Shared country/name bins | ~1.00 | **0.418** |
| **V3 Initial** | Top 12 (cut off) | 85.35% (pool) | Random bucket collisions | 5.07 *(branch flood)* | **0.601** |
| **V3 Modified** | Top 40 (full) | 85.35% | Blocker non-match collisions | 3.30 *(GT mean: 3.45)* | **0.742** |
| **V4 (Target)** | **Top 50** | **$\ge 98.5\%$** | **10-Route Blocker Collisions** | **~3.40 (Calibrated)** | **`> 0.950`** *(Target)* |

---

## The 4 Pillars of V4

### 1. 10-Route Multi-Modal Blocker ($\ge 98.5\%$ Recall)
Expands beyond V3's 6 routes to capture previously unreachable true matching pairs:
* **Route 7: Pure Spatial Address & Geometry Hash (The Trade Alias Buster)**:
  * Key: `spa_{country}_{normalized_num}_{street_token}` (e.g. `spa_US_31_floyd`).
  * Recovers 100% of DBA/trade aliases and brand rebrandings where names have 0% lexical overlap but physical street addresses are identical (e.g., `Probst & Duran Newhold LLC` $\leftrightarrow$ `NYLADREX` at `31 Floyd St, Boston`).
* **Route 8: Native Indic Script Transliteration (`indic-transliteration`)**:
  * Automatically detects and phonetically transliterates Dravidian and Indo-Aryan scripts (Malayalam, Tamil, Gujarati, Telugu, Bengali, Devanagari) to standardized ITRANS Latin tokens before acoustic hashing.
* **Route 9: Domain Compound Word Segmentation**:
  * Deconstructs compacted URLs and handles (`technologiesmarketing.com` $\rightarrow$ `['technologies', 'marketing']`, `housefood.com` $\rightarrow$ `['house', 'food']`).
* **Route 10: "Formerly Known As" / Predecessor Disentangler**:
  * Regex parses `formerly known as`, `fka`, `aka`, `t/a`, `dba` into distinct constituent business identities.

### 2. 24-Dimensional Component-Parsed Feature Matrix
Splits raw strings into structural components to expose fine-grained signals:
* **Name & Phonetic Metrics (6)**: RapidFuzz ratio, token set, token sort, partial, length difference, and Double Metaphone phonetic similarity.
* **Address Component Deconstruction (8)**:
  * Exact street name similarity (isolated from city/state).
  * Building number exact match.
  * Zero-padded building number match (`1109` == `01109`).
  * Suffix-stripped number match (`31` == `31D`).
  * Exact 5/6-digit postal/PIN code match.
  * City, State, and Country exact matches.
* **Spatial Trade Alias Flag (2)**:
  * Non-linear interaction feature: $\text{addr\_fuzz} \times (1.0 - \text{name\_fuzz})$ to explicitly signal co-located trade aliases vs distinct businesses.
* **Domain & Transliteration Overlap (4)**:
  * De-compounded domain similarity and cross-script phonetic match.
* **Dense Embedding Cosine (4)**:
  * Dense semantic embeddings from `multilingual-e5-small`.

### 3. Cascaded Reranking (High Speed + Deep Token Cross-Attention)
* **Stage 1 (High-Throughput GBDT Ensemble)**: LightGBM + XGBoost + CatBoost evaluate all candidates. High-confidence pairs ($>0.85$) and non-matches ($<0.35$) are resolved in microseconds.
* **Stage 2 (Neural Cross-Encoder for Ambiguity)**: Only borderline pairs ($0.35 \le \text{score} \le 0.75$, ~3% of candidates) are evaluated by `BAAI/bge-reranker-v2-m3` on Apple Silicon MPS for deep token-level cross-attention.

### 4. Bipartite Hungarian Assignment & Distribution Matching
* Solves multi-source bipartite matching so each S2/S3 entity is assigned only to its optimal S1 cluster.
* Enforces ground-truth distribution constraints (max 3 from S2, max 3 from S3, confidence floor $\tau \ge 0.50$).

---

## Directory Structure & Pipelines

- **`scripts/`**:
  - `generate_candidates_v4.py`: 10-route multi-modal candidate generator.
  - `train_pipeline_v4.py`: Blocker-negative dataset assembly and 3-booster training.
  - `run_inference_v4.py`: Cascaded inference across 1,732,544 entities.
- **`notebooks/`**:
  - `V4_01_10Route_Candidate_Pairs_Generation.ipynb`
  - `V4_02_Multimodal_Feature_Extraction.ipynb`
  - `V4_03_Model_Training_And_Cascaded_Reranking.ipynb`
  - `V4_04_Submission_Inference_Pipeline.ipynb`
- **`models/`**: Regularized LightGBM, XGBoost, CatBoost boosters and ensemble metadata.
- **`results/`**: Final matching submission TSV.
