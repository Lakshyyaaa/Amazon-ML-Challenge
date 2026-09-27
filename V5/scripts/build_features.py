"""Compute pair features for a candidate file and cache them.

python build_features.py --split train --tag tr12   -> cache/feat_train_tr12.parquet (+label)
"""
import argparse
import os
import pickle
import time

import numpy as np
import pandas as pd

from common import CACHE, DATA
from features import Table, set_admin, attach, build_idf, build_stats, compute
from run_blocking import load_targets


def get_idf(split, tg, col='core'):
    path = f'{CACHE}/idf_{split}.pkl' if col == 'core' else f'{CACHE}/idf_{col}_{split}.pkl'
    if os.path.exists(path):
        return pickle.load(open(path, 'rb'))
    idf = build_idf([tg], col)
    pickle.dump(idf, open(path, 'wb'))
    return idf


def with_norms(df, norms):
    m = norms.set_index('id').bnorm
    df = df.copy()
    df['bnorm'] = m.reindex(df.id.astype(str).values).fillna(0).values.astype(np.float32)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', default='train')
    ap.add_argument('--tag', required=True)
    ap.add_argument('--suffix', default='')
    ap.add_argument('--fold', type=int, default=-1, help='only S1 in this fold; output tag f{fold}')
    ap.add_argument('--k', type=int, default=50)
    ap.add_argument('--chunk', type=int, default=4_000_000)
    a = ap.parse_args()
    t0 = time.time()
    c = pd.read_parquet(f'{CACHE}/cands_{a.split}_{a.tag}.parquet')
    c = c[c.brank < a.k].reset_index(drop=True)
    out_tag = a.tag
    if a.fold >= 0:
        from run_blocking import fold_of
        u = pd.unique(c.s1)
        keep = set(u[fold_of(u) == a.fold])
        c = c[c.s1.isin(keep)].reset_index(drop=True)
        out_tag = f'f{a.fold}'
    s1 = pd.read_parquet(f'{CACHE}/{a.split}_s1.parquet')
    s1 = s1[s1.id.isin(set(c.s1))]
    tg = load_targets(a.split, arrow=True)
    set_admin(a.split)
    idf = get_idf(a.split, tg)
    aidf = get_idf(a.split, tg, 'addr')
    stats = build_stats(pd.read_parquet(f'{CACHE}/{a.split}_s1.parquet', dtype_backend='pyarrow'), tg)
    norms = pd.read_parquet(f'{CACHE}/norms_{a.split}.parquet')
    tg = Table(with_norms(tg, norms))
    s1 = Table(with_norms(s1, norms))
    print(f'{len(c):,} pairs; loaded in {time.time() - t0:.0f}s', flush=True)
    # chunk on S1 boundaries so group features stay correct
    s1_order = c.s1.values
    bounds = [0]
    while bounds[-1] < len(c):
        e = min(bounds[-1] + a.chunk, len(c))
        while e < len(c) and s1_order[e] == s1_order[e - 1]:
            e += 1
        bounds.append(e)
    parts = []
    for lo, hi in zip(bounds, bounds[1:]):
        P = attach(c.iloc[lo:hi], s1, tg)
        F = compute(P, idf, stats, aidf)
        F.insert(0, 's1', P.s1.values)
        F.insert(1, 'tid', P.tid.values)
        parts.append(F)
        print(f'  {hi:,}/{len(c):,} pairs featurized ({time.time() - t0:.0f}s)', flush=True)
    F = pd.concat(parts, ignore_index=True)
    if a.split == 'train':
        gt = pd.read_csv(f'{DATA}/train/train_ground_truth.tsv', sep='\t', dtype=str)
        gt = gt[gt.source1_entity_id.isin(set(c.s1))].dropna()
        gt = gt.assign(tid=gt.matched_entity_ids.str.split(',')).explode('tid')
        pos = set(zip(gt.source1_entity_id, gt.tid))
        F['label'] = np.fromiter(((s, t) in pos for s, t in zip(F.s1, F.tid)), dtype=np.int8, count=len(F))
    F.to_parquet(f'{CACHE}/feat_{a.split}_{out_tag}{a.suffix}.parquet', index=False)
    print(f'saved {len(F):,} rows x {F.shape[1]} cols in {time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
