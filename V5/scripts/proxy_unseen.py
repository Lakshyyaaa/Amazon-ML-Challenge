"""Transfer proxy for France: train on US rows only, evaluate macro F0.5 on India (fold 0)."""
import lightgbm as lgb
import pandas as pd

from common import CACHE, DATA
from run_blocking import fold_of
from selection import load_gt, macro_f05, select_threshold
from stage1_train import DROP, PARAMS

s1 = pd.read_parquet(f'{CACHE}/train_s1.parquet', columns=['id', 'country'])
us = set(s1.id[s1.country == 'US'])
tr = pd.concat([pd.read_parquet(f'{CACHE}/feat_train_f{f}.parquet') for f in (1, 2)])
tr = tr[tr.s1.isin(us)]
es = pd.read_parquet(f'{CACHE}/feat_train_f19.parquet')
es = es[es.s1.isin(us)]
feats = [c for c in tr.columns if c not in DROP]
m = lgb.train(PARAMS, lgb.Dataset(tr[feats], tr.label), 3000,
              valid_sets=[lgb.Dataset(es[feats], es.label)], callbacks=[lgb.early_stopping(100, verbose=False)])
va = pd.read_parquet(f'{CACHE}/feat_train_f0.parquet')
va = va[~va.s1.isin(us)].copy()
va['p'] = m.predict(va[feats], num_threads=10)
va[['s1', 'tid', 'p', 'label']].to_parquet(f'{CACHE}/pred_proxy_india.parquet', index=False)
m.save_model(f'{CACHE}/proxy_us.txt')
imp = pd.Series(m.feature_importance('gain'), feats).sort_values(ascending=False)
print((imp / imp.sum()).round(4).head(12).to_string())
ids = set(s1.id[(fold_of(s1.id) == 0) & (s1.country == 'India')])
gt = load_gt(f'{DATA}/train/train_ground_truth.tsv', ids)
for tau in (0.75, 0.9, 0.95, 0.98):
    print(f'US-only model on India: tau={tau} F0.5={macro_f05(select_threshold(va, tau=tau), gt):.4f}')
