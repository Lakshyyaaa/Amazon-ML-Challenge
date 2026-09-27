"""Stage 2: re-score pairs using competition context from the whole S1 pool.

Every target belongs to at most one S1, so a target strongly claimed by another S1 is
unlikely to belong to this one. Stage-1 probabilities (cross-fitted on train) are turned into
S1-side and target-side context features and a second LightGBM is trained on them.

  python stage2.py pred1 --split train     # p1 for all 20 train folds (A trained on 1-4 / B on 5-8)
  python stage2.py pred1 --split test
  python stage2.py train                   # train on folds 5-12, evaluate on fold 0
"""
import argparse
import os
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from common import CACHE, DATA, V5
from selection import load_gt, macro_f05, select_threshold

KEEP = ['t_addr_empty', 's_addr_empty', 'core_tset', 'core_ratio', 'addr_tset', 'house_eq',
        'house_logdiff', 'brank', 'bscore', 'is_s2', 'name_ratio', 'freq_s1_core_s_core',
        'freq_tg_core_s_core', 'freq_tg_core_t_core', 'freq_s1_core_t_core', 'tok_extra_t']
TRAIN_FOLDS = list(range(9, 17))
VAL_FOLD = 0
P1_MIN = 0.003


def id2int(ids):
    s = pd.Series(ids)
    src = s.str[1].astype(np.int64)
    return src * 10_000_000_000 + s.str[3:].astype(np.int64)


def int2id(x):
    return [f'S{v // 10_000_000_000}-{v % 10_000_000_000}' for v in x]


def compact(F, m, fold):
    feats = m.feature_name()
    out = pd.DataFrame({'s1': id2int(F.s1.values).values, 'tid': id2int(F.tid.values).values,
                        'p1': m.predict(F[feats], num_threads=10).astype(np.float32)})
    for c in KEEP:
        out[c] = F[c].values.astype(np.float32)
    if 'label' in F:
        out['label'] = F.label.values
    out['fold'] = np.int8(fold)
    return out


def pred1(split, chunk=None):
    A = lgb.Booster(model_file=f'{V5}/models/s1A.txt')
    B = lgb.Booster(model_file=f'{V5}/models/s1B.txt')
    t0 = time.time()
    if split == 'train':
        parts = []
        for f in range(20):
            F = pd.read_parquet(f'{CACHE}/feat_train_f{f}.parquet')
            parts.append(compact(F, B if f in (1, 2, 3, 4) else A, f))
            print(f'  fold {f} done ({time.time() - t0:.0f}s)', flush=True)
        out = pd.concat(parts, ignore_index=True)
    else:
        from predict_test import score_test_features
        # model A only: matches how stage 2 saw p1 in training (one model's prediction per pair)
        out = score_test_features(lambda F: compact(F, A, -1))
    out.to_parquet(f'{CACHE}/pred1_{split}.parquet', index=False)
    print(f'saved pred1_{split}: {len(out):,} rows ({time.time() - t0:.0f}s)')


def context(d):
    """Add S1-side and target-side competition features in place."""
    p = d.p1.values
    g1 = d.groupby('s1', sort=False).p1
    d['s1_max'] = g1.transform('max').values
    d['s1_gap'] = p - d.s1_max.values
    d['s1_sum'] = g1.transform('sum').values
    d['s1_cnt50'] = (d.p1 > 0.5).groupby(d.s1, sort=False).transform('sum').values.astype(np.float32)
    d['s1_rank'] = g1.rank(ascending=False, method='first').values.astype(np.float32)
    d['s1_src_rank'] = d.groupby(['s1', 'is_s2'], sort=False).p1.rank(ascending=False, method='first').values.astype(np.float32)
    gt_ = d.groupby('tid', sort=False).p1
    d['t_n'] = gt_.transform('size').values.astype(np.float32)
    d['t_rank'] = gt_.rank(ascending=False, method='first').values.astype(np.float32)
    top = gt_.transform('max').values
    # second-best claim for this target
    d['_p_mask'] = np.where(d.t_rank.values == 1, -1.0, p)
    second = d.groupby('tid', sort=False)._p_mask.transform('max').values
    d['t_other_max'] = np.where(d.t_rank.values == 1, np.maximum(second, 0), top)
    d['t_gap'] = p - d.t_other_max.values
    d['t_sum_other'] = gt_.transform('sum').values - p
    d.drop(columns=['_p_mask'], inplace=True)
    # how contested is this S1's candidate set overall
    lost = ((d.p1 > 0.5) & (d.t_gap < 0)).astype(np.float32)
    d['s1_lost'] = lost.groupby(d.s1, sort=False).transform('sum').values
    return d


FEATS2 = ['p1', 's1_max', 's1_gap', 's1_sum', 's1_cnt50', 's1_rank', 's1_src_rank', 't_n', 't_rank',
          't_other_max', 't_gap', 't_sum_other', 's1_lost'] + KEEP


def train(sib=False):
    t0 = time.time()
    # pairs below P1_MIN can never be selected; drop them before building context (memory)
    d = pd.read_parquet(f'{CACHE}/pred1_train.parquet', filters=[('p1', '>', P1_MIN)])
    print(f'{len(d):,} pairs above P1_MIN ({time.time() - t0:.0f}s)', flush=True)
    d = context(d)
    feats = FEATS2
    if sib:
        from siblings import SIB, add_siblings
        d = add_siblings(d, 'train')
        feats = FEATS2 + SIB
    print(f'context built ({time.time() - t0:.0f}s)', flush=True)
    tr = d[d.fold.isin(TRAIN_FOLDS)]
    es = d[d.fold == 19]
    va = d[d.fold == VAL_FOLD].copy()
    del d
    params = dict(objective='binary', learning_rate=0.05, num_leaves=63, min_data_in_leaf=200,
                  feature_fraction=0.9, bagging_fraction=0.8, bagging_freq=1, num_threads=10, verbose=-1)
    m = lgb.train(params, lgb.Dataset(tr[feats], tr.label), 3000,
                  valid_sets=[lgb.Dataset(es[feats], es.label)],
                  callbacks=[lgb.log_evaluation(500), lgb.early_stopping(100)])
    tag = 'stage2_sib' if sib else 'stage2'
    m.save_model(f'{V5}/models/{tag}.txt')
    va['p2'] = m.predict(va[feats], num_threads=10)
    imp = pd.Series(m.feature_importance('gain'), feats).sort_values(ascending=False)
    print((imp / imp.sum()).round(4).head(15).to_string())

    s1 = pd.read_parquet(f'{CACHE}/train_s1.parquet', columns=['id'])
    from run_blocking import fold_of
    val_ids = set(s1.id[fold_of(s1.id) == VAL_FOLD])
    gt = load_gt(f'{DATA}/train/train_ground_truth.tsv', val_ids)
    va['s1'] = int2id(va.s1.values)
    va['tid'] = int2id(va.tid.values)
    va[['s1', 'tid', 'p1', 'p2', 'label']].to_parquet(f'{CACHE}/pred2_val{"_sib" if sib else ""}.parquet', index=False)
    for col in ('p1', 'p2'):
        best = max((macro_f05(select_threshold(va, prob=col, tau=t), gt), t)
                   for t in (0.4, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85))
        print(f'{col}: best macro F0.5 = {best[0]:.4f} at tau={best[1]}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd')
    ap.add_argument('--split', default='train')
    ap.add_argument('--sib', action='store_true')
    a = ap.parse_args()
    if a.cmd == 'pred1':
        pred1(a.split)
    elif a.cmd == 'train':
        train(a.sib)
