"""Sibling-consensus features for stage 2.

For every candidate r of an S1, compare r with the S1's other candidates:
  confident siblings C (p1 >= CONF): best name similarity, exact name / core agreement, best address
  similarity, house-number agreement, |C|;
  uncertain siblings V (p1 < CONF): best name similarity and how many look alike (decoys come in groups).
A no-address record whose name equals the confirmed siblings' names is almost surely a true match;
a record adding a word/legal form none of the siblings carry is likely a decoy.
"""
import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.process import cpdist

from common import CACHE
from features import Table
from run_blocking import load_targets

CONF = 0.9
SIB = ['sib_n', 'sib_name_max', 'sib_name_eq', 'sib_core_eq', 'sib_addr_max', 'sib_house_eq',
       'sib_legal_new', 'unc_name_max', 'unc_cnt90', 'unc_house_eq_s']


def _legal(name):
    from features import legal_mask
    return legal_mask(name)


def add_siblings(d, split, chunk_pairs=6_000_000):
    """d: stage-2 frame (s1, tid int codes, p1, ...). Adds SIB columns in place, returns d."""
    d = d.sort_values(['s1', 'p1'], ascending=[True, False]).reset_index(drop=True)
    tg = Table(load_targets(split, arrow=True))
    cols = tg.take(d.tid.values, ['name', 'core', 'addr', 'house'], '')
    name, core, addr, house = cols['name'], cols['core'], cols['addr'], cols['house']
    legal = np.fromiter((_legal(x) for x in name), dtype=np.int64, count=len(name))
    s1raw = pd.read_parquet(f'{CACHE}/{split}_s1.parquet', columns=['id', 'house'])
    s1house = dict(zip(pd.Series(s1raw.id).str[3:].astype(np.int64).values + 10_000_000_000, s1raw.house.values))
    s_house = np.array([s1house.get(x, '') for x in d.s1.values], dtype=object)

    conf = d.p1.values >= CONF
    s1 = d.s1.values
    starts = np.flatnonzero(np.r_[True, s1[1:] != s1[:-1]])
    ends = np.r_[starts[1:], len(s1)]
    n = len(d)
    out = {k: np.zeros(n, np.float32) for k in SIB}
    out['sib_addr_max'][:] = -1
    # all ordered pairs (i, j), i != j, within each S1 group, processed in chunks
    sizes = ends - starts
    g0 = 0
    while g0 < len(starts):
        g1, tot = g0, 0
        while g1 < len(starts) and tot < chunk_pairs:
            tot += sizes[g1] * (sizes[g1] - 1)
            g1 += 1
        ii, jj = [], []
        for a, b in zip(starts[g0:g1], ends[g0:g1]):
            if b - a < 2:
                continue
            r = np.arange(a, b)
            I, J = np.meshgrid(r, r, indexing='ij')
            m = I != J
            ii.append(I[m]); jj.append(J[m])
        g0 = g1
        if not ii:
            continue
        ii, jj = np.concatenate(ii), np.concatenate(jj)
        nr = cpdist(name[ii].tolist(), name[jj].tolist(), scorer=fuzz.ratio, workers=10).astype(np.float32)
        both = (addr[ii] != '') & (addr[jj] != '')
        ar = np.full(len(ii), -1, np.float32)
        if both.any():
            ar[both] = cpdist(addr[ii[both]].tolist(), addr[jj[both]].tolist(),
                              scorer=fuzz.token_set_ratio, workers=10)
        name_eq = name[ii] == name[jj]
        core_eq = core[ii] == core[jj]
        h_eq = (house[ii] == house[jj]) & (house[ii] != '')
        new_legal = (legal[ii] & ~legal[jj]) != 0
        cj = conf[jj]
        df = pd.DataFrame({'i': ii, 'c': cj, 'nr': nr, 'ar': ar, 'ne': name_eq, 'ce': core_eq, 'he': h_eq,
                           'nl': new_legal, 'hs': (house[jj] == s_house[ii]) & (house[jj] != '')})
        c = df[df.c].groupby('i')
        agg = c.agg(n=('nr', 'size'), nm=('nr', 'max'), ne=('ne', 'max'), ce=('ce', 'max'), am=('ar', 'max'),
                    he=('he', 'max'), nl=('nl', 'min'))
        idx = agg.index.values
        out['sib_n'][idx] = agg['n'].values
        out['sib_name_max'][idx] = agg['nm'].values / 100
        out['sib_name_eq'][idx] = agg['ne'].values
        out['sib_core_eq'][idx] = agg['ce'].values
        out['sib_addr_max'][idx] = agg['am'].values / 100
        out['sib_house_eq'][idx] = agg['he'].values
        out['sib_legal_new'][idx] = agg['nl'].values  # r has a legal form that NO confident sibling has
        u = df[~df.c]
        u = u.assign(l90=u.nr >= 90, hs90=(u.nr >= 90) & u.hs)
        ua = u.groupby('i').agg(m=('nr', 'max'), k=('l90', 'sum'), h=('hs90', 'sum'))
        idx = ua.index.values
        out['unc_name_max'][idx] = ua['m'].values / 100
        out['unc_cnt90'][idx] = ua['k'].values
        out['unc_house_eq_s'][idx] = ua['h'].values
    for k, v in out.items():
        d[k] = v
    return d
