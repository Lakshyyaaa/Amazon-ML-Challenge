"""Feasibility of 2-hop (target -> target) candidate expansion on validation fold 0.

Queries = targets that stage 2 matched confidently for fold-0 S1s; targets = all train targets.
Measures how many blocker-missed true matches appear among the confident matches' neighbors,
and how many wrong candidates the expansion would add.
"""
import numpy as np
import pandas as pd

from blocker import block
from common import CACHE, DATA
from run_blocking import load_targets
from selection import load_gt

KN = 20
va = pd.read_parquet(f'{CACHE}/pred2_val.parquet')
conf = va[va.p2 >= 0.9]
gt = load_gt(f'{DATA}/train/train_ground_truth.tsv', set(va.s1))
cands = pd.read_parquet(f'{CACHE}/cands_train_f0.parquet').groupby('s1').tid.agg(set).to_dict()
missed = {(s, t) for s, g in gt.items() for t in g if t not in cands.get(s, set())}
print('blocker-missed positives:', len(missed))

tg = load_targets('train')
qids = set(conf.tid)
out = []
for c in ('India', 'US'):
    t = tg[tg.country == c].reset_index(drop=True)
    q = t[t.id.isin(qids)].reset_index(drop=True)
    r = block(q, t, k=KN + 1)
    r['q'] = q.id.values[r.q.values]
    r['t'] = t.id.values[r.t.values]
    out.append(r[r.q != r.t])
nb = pd.concat(out)
nb = nb[nb.brank <= KN]
m = conf[['s1', 'tid']].merge(nb.rename(columns={'q': 'tid', 't': 'nb'}), on='tid')
new = m[['s1', 'nb']].drop_duplicates()
new = new[[n not in cands.get(s, set()) for s, n in zip(new.s1, new.nb)]]
hit = sum((s, n) in missed for s, n in zip(new.s1, new.nb))
print(f'2-hop adds {len(new):,} new pairs ({len(new) / len(gt):.1f}/S1); recovers {hit:,} of {len(missed):,} missed '
      f'({hit / max(len(missed), 1):.1%})')
for k in (3, 5, 10):
    sub = m[m.brank < k][['s1', 'nb']].drop_duplicates()
    sub = sub[[n not in cands.get(s, set()) for s, n in zip(sub.s1, sub.nb)]]
    h = sum((s, n) in missed for s, n in zip(sub.s1, sub.nb))
    print(f'  top-{k} neighbors: +{len(sub):,} pairs, recovers {h:,}')
