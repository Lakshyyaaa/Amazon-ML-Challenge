"""Which features shift on France? Classifier France-test vs train(US+India) on plausible pairs."""
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from common import CACHE, V5
from stage1_train import DROP

A = lgb.Booster(model_file=f'{V5}/models/s1A.txt')
feats = A.feature_name()
fr = pd.read_parquet(f'{CACHE}/feat_test_france.parquet')
tr = pd.concat([pd.read_parquet(f'{CACHE}/feat_train_f{f}.parquet') for f in (5, 6)], ignore_index=True)
for d in (fr, tr):
    d['p1'] = A.predict(d[feats], num_threads=10)
fr, tr = fr[fr.p1 > 0.2], tr[tr.p1 > 0.2]
n = min(len(fr), len(tr), 1_000_000)
X = pd.concat([fr.sample(n, random_state=0), tr.sample(n, random_state=0)], ignore_index=True)
y = np.r_[np.ones(n), np.zeros(n)]
idx = np.random.RandomState(0).permutation(len(X))
cut = int(0.8 * len(X))
a, b = idx[:cut], idx[cut:]
m = lgb.train(dict(objective='binary', learning_rate=0.1, num_leaves=63, verbose=-1, num_threads=10),
              lgb.Dataset(X.iloc[a][feats], y[a]), 200)
print(f'France vs train separability AUC: {roc_auc_score(y[b], m.predict(X.iloc[b][feats])):.4f}')
imp = pd.Series(m.feature_importance('gain'), feats).sort_values(ascending=False)
top = (imp / imp.sum()).head(15)
imp_a = pd.Series(A.feature_importance('gain'), feats)
imp_a = imp_a / imp_a.sum()
print(f"{'feature':24s} {'shift':>7s} {'model imp':>9s} {'France mean':>11s} {'train mean':>10s}")
for f, v in top.items():
    print(f'{f:24s} {v:7.3f} {imp_a[f]:9.3f} {fr[f].mean():11.3f} {tr[f].mean():10.3f}')
print('\nshare of plausible pairs in uncertain band 0.2-0.8: France', round(((fr.p1 < .8)).mean(), 3),
      'train', round(((tr.p1 < .8)).mean(), 3))
