"""Reproduce test conditions on train: drop a fraction of S1 entities so their S2/S3 records
become orphans (test has ~5.8 targets/S1 vs 4.67 in train).

1. Score validation fold 0 with the existing stage-2 model under orphan conditions.
2. Retrain stage 2 under orphan conditions and score again.
"""
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd

from common import CACHE, DATA, V5
from selection import load_gt, macro_f05, select_threshold
from stage2 import FEATS2, P1_MIN, TRAIN_FOLDS, context, int2id

DROP_FRAC = float(sys.argv[1]) if len(sys.argv) > 1 else 0.195


def dropped(s1_int):
    return ((s1_int * 2654435761) % 1_000_003) / 1_000_003 < DROP_FRAC


d = pd.read_parquet(f'{CACHE}/pred1_train.parquet', filters=[('p1', '>', P1_MIN)])
d = d[~dropped(d.s1.values)].reset_index(drop=True)
d = context(d)
va = d[d.fold == 0].copy()
s1 = pd.read_parquet(f'{CACHE}/train_s1.parquet', columns=['id'])
from run_blocking import fold_of
ids = s1.id[fold_of(s1.id) == 0]
num = ids.str[3:].astype(np.int64).values + 10_000_000_000
gt = load_gt(f'{DATA}/train/train_ground_truth.tsv', set(ids[~dropped(num)]))
va['s1s'], va['tids'] = int2id(va.s1.values), int2id(va.tid.values)


def score(p, tag):
    v = va[['s1s', 'tids']].rename(columns={'s1s': 's1', 'tids': 'tid'}).assign(p=p)
    res = {t: round(macro_f05(select_threshold(v, tau=t), gt), 4) for t in (0.6, 0.7, 0.8, 0.85, 0.9)}
    print(tag, res, flush=True)


old = lgb.Booster(model_file=f'{V5}/models/stage2.txt')
score(va.p1.values, f'drop={DROP_FRAC} stage-1 only:')
score(old.predict(va[FEATS2], num_threads=10), f'drop={DROP_FRAC} current stage-2 model:')
tr, es = d[d.fold.isin(TRAIN_FOLDS)], d[d.fold == 19]
params = dict(objective='binary', learning_rate=0.05, num_leaves=63, min_data_in_leaf=200,
              feature_fraction=0.9, bagging_fraction=0.8, bagging_freq=1, num_threads=10, verbose=-1)
m = lgb.train(params, lgb.Dataset(tr[FEATS2], tr.label), 3000, valid_sets=[lgb.Dataset(es[FEATS2], es.label)],
              callbacks=[lgb.early_stopping(100, verbose=False)])
m.save_model(f'{V5}/models/stage2_orphan.txt')
score(m.predict(va[FEATS2], num_threads=10), f'drop={DROP_FRAC} stage-2 retrained with orphans:')
