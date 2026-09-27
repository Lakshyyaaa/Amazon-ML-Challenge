"""Vectorized IDF-weighted blocker.

Each record becomes a bag of hashed blocking features (name tokens, char 4-grams of the
concatenated core name, name x house number, house number x address words, domain/concat
prefix, acronym x house). Score(q, t) = sum of IDF of shared features, computed as a sparse
matrix product per country. Returns the top-K targets per query.
"""
import math
import zlib
import multiprocessing as mp
Pool = mp.get_context("fork").Pool

import numpy as np
import pandas as pd
import scipy.sparse as sp

NBITS = 26
MASK = (1 << NBITS) - 1
ADDR_STOP = {
    'street', 'road', 'avenue', 'drive', 'lane', 'court', 'place', 'boulevard', 'suite', 'unit',
    'floor', 'apartment', 'near', 'opposite', 'no', 'door', 'house', 'plot', 'flat', 'block',
    'box', 'po', 'pmb', 'city', 'town', 'township', 'rue', 'de', 'du', 'des', 'la', 'le', 'et',
    'main', 'cross', 'north', 'south', 'east', 'west', 'nagar', 'colony', 'sector', 'the', 'of',
    'and', 'dist', 'district', 'post', 'shop', 'building', 'bldg', 'hn', 'h', 'ground', 'first',
}


def _h(s):
    return zlib.crc32(s.encode()) & MASK


def record_features(name, core, dom, alias, addr, nums):
    f = []
    toks = core.split()
    if not toks and name:
        toks = [t for t in name.split() if len(t) > 1]
    for t in toks[:8]:
        f.append('t' + t)
    for a, b in zip(toks, toks[1:]):
        f.append('b' + '_'.join(sorted((a, b))))
    cc = ''.join(toks)
    for i in range(0, max(len(cc) - 3, 0)):
        f.append('g' + cc[i:i + 4])
    if len(cc) >= 6:
        f.append('c' + cc[:8])
    if dom:
        f.append('c' + dom[:8])
        for i in range(0, max(len(dom) - 3, 0)):
            f.append('g' + dom[i:i + 4])
    if alias:
        for p in alias.split('|'):
            ptoks = p.split()
            for t in ptoks[:4]:
                f.append('t' + t)
    aw = [w for w in addr.split() if not w.isdigit() and len(w) >= 3 and w not in ADDR_STOP]
    houses = []
    for n in nums.split():
        if n != '0' and n not in houses:
            houses.append(n)
        if len(houses) == 3:
            break
    if cc:
        f.append('e' + cc)
    for hs in houses:
        f.append('h' + hs)  # weak on its own, pruned by max_df when common
        for t in toks[:4]:
            f.append('nh' + t[:4] + '_' + hs)
        if len(toks) >= 2:
            f.append('ah' + ''.join(t[0] for t in toks[:4]) + '_' + hs)
        for w in aw[:8]:
            f.append('hw' + hs + '_' + w[:5])
    if not houses:
        for a, b in zip(aw[:6], aw[1:7]):
            f.append('aa' + a[:5] + '_' + b[:5])
    for t in toks[:3]:
        for w in aw[:6]:
            f.append('nw' + t[:4] + '_' + w[:5])
    return f


def featurize(df):
    """Return CSR (n x 2^NBITS) binary matrix of hashed features."""
    indptr = [0]
    idx = []
    for name, core, dom, alias, addr, nums in zip(df.name, df.core, df.dom, df.alias, df.addr, df.nums):
        fs = {_h(x) for x in record_features(name, core, dom, alias, addr, nums)}
        idx.extend(fs)
        indptr.append(len(idx))
    idx = np.fromiter(idx, dtype=np.int32, count=len(idx))
    data = np.ones(len(idx), dtype=np.float32)
    return sp.csr_matrix((data, idx, np.asarray(indptr, dtype=np.int64)), shape=(len(df), 1 << NBITS))


def _featurize_par(df, workers=10):
    chunks = [df.iloc[i:i + 100000] for i in range(0, len(df), 100000)]
    with Pool(workers) as p:
        mats = p.map(featurize, chunks)
    return sp.vstack(mats, format='csr')


_G = {}


def _topk_chunk(args):
    lo, hi, k = args
    Q = _G['Q'][lo:hi]
    S = (Q @ _G['TT']).tocsr()
    out_q, out_t, out_s = [], [], []
    for r in range(S.shape[0]):
        a, b = S.indptr[r], S.indptr[r + 1]
        if a == b:
            continue
        d = S.data[a:b]
        ix = S.indices[a:b]
        if b - a > k:
            sel = np.argpartition(-d, k)[:k]
            d, ix = d[sel], ix[sel]
        o = np.argsort(-d, kind='stable')
        out_q.append(np.full(len(o), lo + r, dtype=np.int32))
        out_t.append(ix[o].astype(np.int32))
        out_s.append(d[o])
    if not out_q:
        return (np.zeros(0, np.int32),) * 2 + (np.zeros(0, np.float32),)
    return np.concatenate(out_q), np.concatenate(out_t), np.concatenate(out_s)


def block(queries, targets, k=50, max_df=2000, workers=10, log=print):
    """queries/targets: normalized DataFrames (same country). Returns DataFrame(q, t, bscore, brank)
    with positional indices into the given frames."""
    T = _featurize_par(targets, workers)
    df = np.bincount(T.indices, minlength=1 << NBITS)
    n = T.shape[0]
    idf = np.log((n + 1) / (df + 1)).astype(np.float32)
    idf[df > max_df] = 0
    idf[df == 0] = 0
    Q = _featurize_par(queries, workers)
    Q.data = idf[Q.indices]
    Q.eliminate_zeros()
    log(f'  featurized: {Q.shape[0]:,} queries, {n:,} targets, nnz T={T.nnz:,}')
    _G['Q'] = Q
    _G['TT'] = T.T.tocsr()
    step = 500
    jobs = [(i, min(i + step, Q.shape[0]), k) for i in range(0, Q.shape[0], step)]
    with Pool(workers) as p:
        res = p.map(_topk_chunk, jobs, chunksize=1)
    _G.clear()
    q = np.concatenate([r[0] for r in res])
    t = np.concatenate([r[1] for r in res])
    s = np.concatenate([r[2] for r in res])
    out = pd.DataFrame({'q': q, 't': t, 'bscore': s})
    out['brank'] = out.groupby('q').cumcount().astype(np.int16)
    return out
