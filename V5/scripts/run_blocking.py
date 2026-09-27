"""Run the blocker for a split.

train: queries = S1 rows in the requested folds (fold = crc32(id) % 20), targets = all train S2+S3.
test:  queries = all test S1, targets = all test S2+S3.
Output: V5/cache/cands_{split}_{tag}.parquet with columns s1 (id), tid (id), bscore, brank.
"""
import argparse
import time
import zlib

import numpy as np
import pandas as pd

from blocker import block
from common import CACHE, DATA


def fold_of(ids):
    return np.array([zlib.crc32(i.encode()) % 20 for i in ids])


def load_targets(split, arrow=False):
    kw = {'dtype_backend': 'pyarrow'} if arrow else {}
    t2 = pd.read_parquet(f'{CACHE}/{split}_s2.parquet', **kw)
    t3 = pd.read_parquet(f'{CACHE}/{split}_s3.parquet', **kw)
    return pd.concat([t2, t3], ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', default='train')
    ap.add_argument('--folds', default='0')
    ap.add_argument('--tag', default='val')
    ap.add_argument('--k', type=int, default=80)
    ap.add_argument('--max_df', type=int, default=2000)
    a = ap.parse_args()

    s1 = pd.read_parquet(f'{CACHE}/{a.split}_s1.parquet')
    if a.split == 'train':
        folds = [int(x) for x in a.folds.split(',')]
        s1 = s1[np.isin(fold_of(s1.id), folds)].reset_index(drop=True)
    tg = load_targets(a.split)
    outs = []
    for c in sorted(s1.country.unique()):
        t0 = time.time()
        q = s1[s1.country == c].reset_index(drop=True)
        t = tg[tg.country == c].reset_index(drop=True)
        print(f'[{c}] {len(q):,} queries x {len(t):,} targets', flush=True)
        r = block(q, t, k=a.k, max_df=a.max_df)
        r['s1'] = q.id.values[r.q.values]
        r['tid'] = t.id.values[r.t.values]
        outs.append(r[['s1', 'tid', 'bscore', 'brank']])
        print(f'[{c}] done in {time.time() - t0:.0f}s, {len(r):,} pairs', flush=True)
    out = pd.concat(outs, ignore_index=True)
    path = f'{CACHE}/cands_{a.split}_{a.tag}.parquet'
    out.to_parquet(path, index=False)
    print('saved', path, len(out))

    if a.split == 'train':
        gt = pd.read_csv(f'{DATA}/train/train_ground_truth.tsv', sep='\t', dtype=str)
        gt = gt[gt.source1_entity_id.isin(set(s1.id))].dropna()
        gt = gt.assign(tid=gt.matched_entity_ids.str.split(',')).explode('tid')
        pos = set(zip(gt.source1_entity_id, gt.tid))
        hit = np.array([(s, t) in pos for s, t in zip(out.s1, out.tid)])
        print(f'total positives {len(pos):,}')
        for k in (5, 10, 15, 20, 30, 40, 50, 60, 80, 100):
            if k <= a.k:
                print(f'  recall@{k}: {hit[out.brank.values < k].sum() / len(pos):.4f}')


if __name__ == '__main__':
    main()
