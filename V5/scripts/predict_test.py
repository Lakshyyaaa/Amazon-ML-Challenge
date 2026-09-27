"""Score test candidates and write the submission files.

  python predict_test.py --model stage1 --tau 0.75     # single-stage (stage-1 model only)
  python predict_test.py --stage2 --tau 0.7            # uses cache/pred1_test.parquet + models/stage2.txt

Requires cache/test_s*.parquet (prep.py) and cache/cands_test_test.parquet
(run_blocking.py --split test --tag test --k 50).
Writes V5/output/matching_results.tsv and V5/output/candidate_pairs.tsv.
"""
import argparse
import os
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from build_features import get_idf, with_norms
from common import CACHE, V5
from features import Table, set_admin, attach, build_stats, compute
from run_blocking import load_targets
from selection import select_threshold

K = 50


def test_cands():
    c = pd.read_parquet(f'{CACHE}/cands_test_test.parquet')
    return c[c.brank < K].reset_index(drop=True)


def s1_bounds(codes, chunk):
    bounds = [0]
    while bounds[-1] < len(codes):
        e = min(bounds[-1] + chunk, len(codes))
        while e < len(codes) and codes[e] == codes[e - 1]:
            e += 1
        bounds.append(e)
    return bounds


def score_test_features(fn, chunk=2_000_000, countries=None):
    """Featurize all test candidate pairs chunk by chunk (S1-aligned) and concat fn(F) results.
    Memory-lean: candidates stay in Arrow, strings are materialized per chunk only."""
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    t0 = time.time()
    tbl = pq.read_table(f'{CACHE}/cands_test_test.parquet', columns=['s1', 'tid', 'bscore', 'brank'])
    tbl = tbl.filter(pc.less(tbl['brank'], K))
    if countries is not None:
        s1c = pd.read_parquet(f'{CACHE}/test_s1.parquet', columns=['id', 'country'])
        import pyarrow as pa
        keep = pa.array(s1c.id[s1c.country.isin(countries)].tolist())
        tbl = tbl.filter(pc.is_in(tbl['s1'], value_set=keep))
    codes = pc.dictionary_encode(tbl['s1']).combine_chunks().indices.to_numpy()
    s1 = pd.read_parquet(f'{CACHE}/test_s1.parquet', dtype_backend='pyarrow')
    tg = load_targets('test', arrow=True)
    set_admin('test')
    idf = get_idf('test', tg)
    aidf = get_idf('test', tg, 'addr')
    stats = build_stats(s1, tg)
    norms = pd.read_parquet(f'{CACHE}/norms_test.parquet')
    s1, tg = Table(with_norms(s1, norms)), Table(with_norms(tg, norms))
    print(f'loaded ({time.time() - t0:.0f}s)', flush=True)
    out = []
    n = tbl.num_rows
    bounds = s1_bounds(codes, chunk)
    for lo, hi in zip(bounds, bounds[1:]):
        P = attach(tbl.slice(lo, hi - lo).to_pandas(), s1, tg)
        F = compute(P, idf, stats, aidf)
        F.insert(0, 's1', P.s1.values)
        F.insert(1, 'tid', P.tid.values)
        out.append(fn(F))
        print(f'  scored {hi:,}/{n:,} ({time.time() - t0:.0f}s)', flush=True)
    return pd.concat(out, ignore_index=True)


def write_submission(sel, tag, name='matching_results.tsv'):
    s1 = pd.read_parquet(f'{CACHE}/test_s1.parquet', columns=['id', 'country'])
    out_dir = f'{V5}/output'
    os.makedirs(out_dir, exist_ok=True)
    with open(f'{out_dir}/{name}', 'w') as f:
        f.write('source1_entity_id\tmatched_entity_ids\n')
        for sid in s1.id.values:
            m_ = sel.get(sid)
            f.write(f"{sid}\t{','.join(sorted(m_)) if m_ else ''}\n")
    cand_path = f'{out_dir}/candidate_pairs.tsv'
    if not os.path.exists(cand_path):
        c = test_cands()
        cand = c.groupby('s1', sort=False).tid.agg(','.join).to_dict()
        with open(cand_path, 'w') as f:
            f.write('source1_entity_id\tcandidate_entity_ids\n')
            for sid in s1.id.values:
                f.write(f"{sid}\t{cand.get(sid, '')}\n")
    print(f'[{tag}] wrote {out_dir}/{name}')
    for ctry in sorted(s1.country.unique()):
        ids = s1.id[s1.country == ctry].values
        k = sum(1 for s in ids if sel.get(s))
        m_ = sum(len(sel.get(s, ())) for s in ids)
        print(f'  {ctry}: S1={len(ids):,} empty={1 - k / len(ids):.3f} avg matches/S1={m_ / len(ids):.2f}')


TRAIN_COUNTRIES = {'US', 'India'}


def select_per_country(pr, tau):
    """Threshold per country. Countries unseen in training (France) get the tau whose average
    matches/S1 equals that of the training countries at `tau` (validated on a US->India proxy:
    this picks the best threshold for an unseen country)."""
    s1 = pd.read_parquet(f'{CACHE}/test_s1.parquet', columns=['id', 'country'])
    cty = dict(zip(s1.id, s1.country))
    n_by = s1.country.value_counts().to_dict()
    pr = pr.assign(c=pr.s1.map(cty))
    seen = pr[pr.c.isin(TRAIN_COUNTRIES)]
    ref = (seen.p >= tau).sum() / sum(n_by[c] for c in TRAIN_COUNTRIES)
    sel = select_threshold(seen, tau=tau)
    for c in sorted(set(n_by) - TRAIN_COUNTRIES):
        d = pr[pr.c == c]
        grid = np.arange(tau, 0.995, 0.005)
        avg = np.array([(d.p >= t).sum() / n_by[c] for t in grid])
        t_c = float(grid[np.argmin(np.abs(avg - ref))])
        print(f'  {c}: tau={t_c:.3f} (avg matches/S1 {avg.min():.2f}..{avg.max():.2f}, reference {ref:.3f})')
        sel.update(select_threshold(d, tau=t_c))
    return sel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='stage1')
    ap.add_argument('--stage2', action='store_true')
    ap.add_argument('--tau', type=float, default=0.75)
    ap.add_argument('--name', default='matching_results.tsv')
    a = ap.parse_args()
    if a.stage2:
        from stage2 import FEATS2, P1_MIN, context, int2id
        d = context(pd.read_parquet(f'{CACHE}/pred1_test.parquet', filters=[('p1', '>', P1_MIN)]))
        m = lgb.Booster(model_file=f'{V5}/models/stage2.txt')
        d['p'] = m.predict(d[FEATS2], num_threads=10)
        d = d[d.p >= a.tau]
        pr = pd.DataFrame({'s1': int2id(d.s1.values), 'tid': int2id(d.tid.values), 'p': d.p.values})
        tag = 'stage2'
    else:
        path = f'{CACHE}/pred_{a.model}_test.parquet'
        if not os.path.exists(path):
            m = lgb.Booster(model_file=f'{V5}/models/{a.model}.txt')
            feats = m.feature_name()
            score_test_features(lambda F: pd.DataFrame(
                {'s1': F.s1.values, 'tid': F.tid.values, 'p': m.predict(F[feats], num_threads=10)})
            ).to_parquet(path, index=False)
        pr = pd.read_parquet(path)
        tag = a.model
    write_submission(select_per_country(pr, a.tau), tag, a.name)


if __name__ == '__main__':
    main()
