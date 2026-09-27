"""Validate self-training for an unseen country on the US -> India proxy.

1. Train on US rows only (folds 1,2).
2. Predict India rows of folds 5-8 (treated as unlabeled), keep confident pairs as pseudo-labels.
3. Retrain on US + pseudo-labeled India, repeat for R rounds.
4. Report India fold-0 macro F0.5 after each round (true labels, never used for training).
"""
import lightgbm as lgb
import numpy as np
import pandas as pd

from common import CACHE, DATA
from run_blocking import fold_of
from selection import load_gt, macro_f05, select_threshold
from stage1_train import DROP, PARAMS

POS, NEG, ROUNDS = 0.98, 0.02, 2

s1 = pd.read_parquet(f'{CACHE}/train_s1.parquet', columns=['id', 'country'])
us = set(s1.id[s1.country == 'US'])
load = lambda fs: pd.concat([pd.read_parquet(f'{CACHE}/feat_train_f{f}.parquet') for f in fs], ignore_index=True)
tr = load([1, 2]); tr = tr[tr.s1.isin(us)]
es = load([19]); es = es[es.s1.isin(us)]
un = load([5, 6, 7, 8]); un = un[~un.s1.isin(us)].reset_index(drop=True)   # unlabeled India
va = load([0]); va = va[~va.s1.isin(us)].reset_index(drop=True)
feats = [c for c in tr.columns if c not in DROP]
ids = set(s1.id[(fold_of(s1.id) == 0) & (s1.country == 'India')])
gt = load_gt(f'{DATA}/train/train_ground_truth.tsv', ids)


def fit(X, y):
    return lgb.train(PARAMS, lgb.Dataset(X, y), 3000, valid_sets=[lgb.Dataset(es[feats], es.label)],
                     callbacks=[lgb.early_stopping(100, verbose=False)])


def report(m, tag):
    p = m.predict(va[feats], num_threads=10)
    d = va[['s1', 'tid']].assign(p=p)
    res = {t: round(macro_f05(select_threshold(d, tau=t), gt), 4) for t in (0.5, 0.65, 0.75, 0.85, 0.9, 0.95)}
    print(tag, res, flush=True)


m = fit(tr[feats], tr.label)
report(m, 'round 0 (US only):')
for r in range(1, ROUNDS + 1):
    p = m.predict(un[feats], num_threads=10)
    sel = (p > POS) | (p < NEG)
    pl = un.loc[sel, feats].assign(label=(p[sel] > POS).astype(np.int8))
    acc = (un.label.values[sel] == pl.label.values).mean()
    print(f'round {r}: {sel.mean():.3f} of unlabeled pairs pseudo-labeled, pos={pl.label.sum():,}, '
          f'pseudo-label accuracy (hidden truth) {acc:.4f}', flush=True)
    X = pd.concat([tr[feats], pl[feats]], ignore_index=True)
    y = np.concatenate([tr.label.values, pl.label.values])
    m = fit(X, y)
    report(m, f'round {r}:')
