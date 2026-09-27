"""Pairwise features for (S1, target) candidate pairs."""
import math
import multiprocessing as mp

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler, Levenshtein
from rapidfuzz.process import cpdist

Pool = mp.get_context('fork').Pool
_G = {}


def _num_feats(a, b):
    """Relations between two house-number strings."""
    if not a or not b:
        return 0.0, 0.0, 0.0, -1.0
    eq = float(a == b)
    pre = float(a != b and (a.startswith(b) or b.startswith(a)))
    ed1 = float(len(a) == len(b) and Levenshtein.distance(a, b) == 1)
    try:
        diff = math.log1p(abs(int(a[:9]) - int(b[:9])))
    except ValueError:
        diff = -1.0
    return eq, pre, ed1, diff


def _token_feats(sc, tc, idf):
    S, T = sc.split(), tc.split()
    if not S or not T:
        return (0.0,) * 9
    matched_t = set()
    w_s = w_m = 0.0
    miss_s = 0
    max_miss = 0.0
    for s in S:
        w = idf.get(s, 12.0)
        w_s += w
        best, bj = 0.0, -1
        for j, t in enumerate(T):
            if s == t:
                best, bj = 1.0, j
                break
            if s[0] == t[0] or s[-1] == t[-1]:
                r = fuzz.ratio(s, t)
                if r > best * 100 and r >= 75:
                    best, bj = r / 100.0, j
        if bj >= 0:
            matched_t.add(bj)
            w_m += w * best
        else:
            miss_s += 1
            max_miss = max(max_miss, w)
    extra = [T[j] for j in range(len(T)) if j not in matched_t]
    w_t = sum(idf.get(t, 12.0) for t in T)
    w_extra = sum(idf.get(t, 12.0) for t in extra)
    max_extra = max((idf.get(t, 12.0) for t in extra), default=0.0)
    return (w_m / max(w_s, 1e-6), w_m / max(w_t, 1e-6), float(miss_s), float(len(extra)),
            w_extra, max_extra, float(len(S)), float(len(T)), max_miss)


LOOP_COLS = ['s_core', 't_core', 's_nums', 't_nums', 's_house', 't_house', 's_postal', 't_postal',
             's_dom', 't_dom', 't_alias', 's_addr', 't_addr', 's_country']
N_LOOP = 28


_ADMIN_RE = {}


def set_admin(split):
    """Load learned admin phrases (admin.py) so address-word features ignore state/region/department."""
    import pickle
    import re
    from admin import build
    _ADMIN_RE.clear()
    for c, ph in build(split).items():
        if ph:
            _ADMIN_RE[c] = re.compile(r'\b(?:' + '|'.join(re.escape(p) for p in ph) + r')\b')


def addr_words(a, country=None):
    r = _ADMIN_RE.get(country)
    if r is not None:
        a = r.sub(' ', a)
    return ' '.join(w for w in a.split() if not w.isdigit() and len(w) >= 2)


def _loop_feats(cols):
    """cols: tuple of equal-length lists (LOOP_COLS order), pickled to the worker so the
    parent's string objects are never touched (avoids copy-on-write blowup under fork)."""
    idf, aidf = _G['idf'], _G['aidf']
    sc, tc, sn, tn, sh, th, sp_, tp_, sd, td, ta, sa, tadr, cty = cols
    out = np.zeros((len(sc), N_LOOP), dtype=np.float32)
    for i in range(len(sc)):
        r = out[i]
        f = _token_feats(sc[i], tc[i], idf)
        r[0:8] = f[:8]
        r[20] = f[8]
        if sa[i] and tadr[i]:
            g = _token_feats(addr_words(sa[i], cty[i]), addr_words(tadr[i], cty[i]), aidf)
            r[21], r[22], r[23], r[24], r[25], r[26] = g[0], g[1], g[2], g[4], g[5], g[8]
            r[27] = 1.0
        a, b = sh[i], th[i]
        r[8:12] = _num_feats(a, b)
        A, B = set(sn[i].split()), set(tn[i].split())
        r[12] = len(A & B)
        r[13] = len(A | B)
        r[14] = float(bool(A) and bool(B) and a in B)  # s house appears anywhere in t numbers
        p, q = sp_[i], tp_[i]
        r[15] = 1.0 if (p and q and p == q) else (-1.0 if (p and q) else 0.0)
        # domain / concatenated name
        s_cc = sc[i].replace(' ', '')
        t_d = td[i] or ''
        if t_d and s_cc:
            r[16] = fuzz.ratio(s_cc, t_d) / 100.0
            r[17] = fuzz.partial_ratio(s_cc, t_d) / 100.0 if len(t_d) >= 3 else 0.0
        elif sd[i] and tc[i]:
            r[16] = fuzz.ratio(sd[i], tc[i].replace(' ', '')) / 100.0
        # acronym: target core is initials of source core
        st = sc[i].split()
        tcs = tc[i].replace(' ', '')
        if len(st) >= 2 and 1 <= len(tcs) <= 5:
            r[18] = float(tcs == ''.join(x[0] for x in st)[:len(tcs)] and len(tcs) >= 2)
        if ta[i]:
            r[19] = max(fuzz.token_set_ratio(sc[i], part) for part in ta[i].split('|')) / 100.0
    return out


LOOP_NAMES = ['tok_cov_s', 'tok_cov_t', 'tok_miss_s', 'tok_extra_t', 'w_extra_t', 'max_extra_idf',
              'ntok_s', 'ntok_t', 'house_eq', 'house_prefix', 'house_ed1', 'house_logdiff',
              'nums_inter', 'nums_union', 'house_in_t', 'postal', 'dom_ratio', 'dom_partial',
              'acronym', 'alias_tset', 'max_miss_idf', 'a_cov_s', 'a_cov_t', 'a_miss_s', 'a_w_extra',
              'a_max_extra', 'a_max_miss', 'a_both']


LEGAL_FAMILY = {
    'pvt': 'private', 'private': 'private', 'ltd': 'limited', 'limited': 'limited',
    'inc': 'inc', 'incorporated': 'inc', 'corp': 'corp', 'corporation': 'corp',
    'co': 'co', 'company': 'co', 'llc': 'llc', 'llp': 'llp', 'plc': 'plc', 'pc': 'pc',
    'public': 'public', 'sarl': 'sarl', 'sas': 'sas', 'sasu': 'sasu', 'eurl': 'eurl',
    'sci': 'sci', 'sa': 'sa', 'snc': 'snc', 'lp': 'lp', 'pllc': 'pllc', 'ei': 'ei',
    'selarl': 'selarl', 'scop': 'scop', 'gie': 'gie', 'scp': 'scp', 'sca': 'sca', 'eirl': 'eirl',
}
FAMILIES = sorted(set(LEGAL_FAMILY.values()))
FAM_BIT = {f: 1 << i for i, f in enumerate(FAMILIES)}


def legal_mask(name):
    s = name.replace('l l c', 'llc').replace('l l p', 'llp')
    m = 0
    for t in s.split():
        f = LEGAL_FAMILY.get(t)
        if f:
            m |= FAM_BIT[f]
    return m


def _legal_feats(sn, tn):
    a = np.fromiter((legal_mask(x) for x in sn), dtype=np.int64, count=len(sn))
    b = np.fromiter((legal_mask(x) for x in tn), dtype=np.int64, count=len(tn))
    pc = lambda v: np.array([bin(x).count('1') for x in v], dtype=np.float32)
    inter, union = pc(a & b), pc(a | b)
    F = {'legal_jacc': np.where(union > 0, inter / np.maximum(union, 1), -1).astype(np.float32),
         'legal_s_only': pc(a & ~b), 'legal_t_only': pc(b & ~a)}
    pv, pu, lim = FAM_BIT['private'], FAM_BIT['public'], FAM_BIT['limited']
    F['legal_priv_pub'] = (((a & pv) > 0) & ((b & pu) > 0) | ((a & pu) > 0) & ((b & pv) > 0)).astype(np.float32)
    F['legal_priv_diff'] = (((a & pv) > 0) != ((b & pv) > 0)).astype(np.float32)
    F['legal_lim_diff'] = (((a & lim) > 0) != ((b & lim) > 0)).astype(np.float32)
    F['s_legal_mask'] = a.astype(np.float32)
    F['t_legal_mask'] = b.astype(np.float32)
    return F


def compute(P, idf, stats=None, aidf=None, workers=10):
    """P: DataFrame with s_*/t_* columns (name, core, dom, alias, addr, house, postal, nums),
    bscore, brank, is_s2. Returns float32 feature DataFrame."""
    F = {}
    for col, a, b, scorers in [
        ('name', 's_name', 't_name', [('ratio', fuzz.ratio)]),
        ('core', 's_core', 't_core', [('ratio', fuzz.ratio), ('tset', fuzz.token_set_ratio),
                                      ('tsort', fuzz.token_sort_ratio), ('partial', fuzz.partial_ratio),
                                      ('jw', JaroWinkler.normalized_similarity)]),
        ('addr', 's_addr', 't_addr', [('ratio', fuzz.ratio), ('tset', fuzz.token_set_ratio),
                                      ('partial', fuzz.partial_ratio)]),
    ]:
        x, y = P[a].tolist(), P[b].tolist()
        for nm, sc in scorers:
            v = cpdist(x, y, scorer=sc, workers=workers).astype(np.float32)
            F[f'{col}_{nm}'] = v / 100.0 if nm != 'jw' else v
    # address without numbers (street/city words only)
    cty = P.s_country.values
    xs = [addr_words(a, c) for a, c in zip(P.s_addr.values, cty)]
    ys = [addr_words(a, c) for a, c in zip(P.t_addr.values, cty)]
    F['addrw_tset'] = cpdist(xs, ys, scorer=fuzz.token_set_ratio, workers=workers).astype(np.float32) / 100
    F['addrw_tsort'] = cpdist(xs, ys, scorer=fuzz.token_sort_ratio, workers=workers).astype(np.float32) / 100
    F['s_addr_empty'] = (P.s_addr.values == '').astype(np.float32)
    F['t_addr_empty'] = (P.t_addr.values == '').astype(np.float32)
    F['s_core_len'] = P.s_core.str.len().values.astype(np.float32)
    F['t_core_len'] = P.t_core.str.len().values.astype(np.float32)
    F['core_eq'] = (P.s_core.values == P.t_core.values).astype(np.float32)
    F['t_nonlatin_hint'] = P.t_name.str.contains(r'\b(?:pvt|pra|li)\b', regex=True).values.astype(np.float32)

    F.update(_legal_feats(P.s_name.values, P.t_name.values))
    if stats is not None:
        for key, col in (('s1_core', 's_core'), ('tg_core', 's_core'), ('tg_core', 't_core'),
                         ('s1_core', 't_core')):
            d = stats[key]
            ks = P.s_country.values + '|' + P[col].values
            F[f'freq_{key}_{col}'] = np.fromiter((d.get(x, 0) for x in ks), dtype=np.float32, count=len(P))

    _G['idf'] = idf
    _G['aidf'] = aidf if aidf is not None else {}
    n = len(P)
    step = 50000
    arrs = [P[c].tolist() for c in LOOP_COLS]
    jobs = (tuple(a[i:i + step] for a in arrs) for i in range(0, n, step))
    with Pool(workers) as p:
        parts = p.map(_loop_feats, jobs, chunksize=1)
    _G.clear()
    L = np.vstack(parts) if parts else np.zeros((0, len(LOOP_NAMES)), np.float32)
    for j, nm in enumerate(LOOP_NAMES):
        F[nm] = L[:, j]
    F['bscore'] = P.bscore.values.astype(np.float32)
    if 's_bnorm' in P:
        sn_, tn_ = P.s_bnorm.values.astype(np.float32), P.t_bnorm.values.astype(np.float32)
        F['bdice'] = 2 * F['bscore'] / np.maximum(sn_ + tn_, 1e-3)
        F['bcov_s'] = F['bscore'] / np.maximum(sn_, 1e-3)
        F['bcov_t'] = F['bscore'] / np.maximum(tn_, 1e-3)
    F['brank'] = P.brank.values.astype(np.float32)
    F['is_s2'] = P.is_s2.values.astype(np.float32)
    F = pd.DataFrame(F)
    # group context over the S1's candidate list
    g = P.s1.values
    grp = pd.Series(F.bscore.values).groupby(g)
    F['bscore_rel'] = F.bscore / grp.transform('max').values
    F['n_cands'] = grp.transform('size').values.astype(np.float32)
    for c in ('core_tset', 'addr_tset', 'core_ratio'):
        mx = pd.Series(F[c].values).groupby(g).transform('max').values
        F[f'{c}_gap'] = F[c].values - mx
    return F


def build_stats(s1, tg):
    """Exact core-name frequencies within the S1 pool and the target pool (per country)."""
    key = lambda df: df.country + '|' + df.core
    return {'s1_core': key(s1).value_counts().to_dict(), 'tg_core': key(tg).value_counts().to_dict()}


def build_idf(target_frames, col='core'):
    from collections import Counter
    c = Counter()
    n = 0
    for df in target_frames:
        vals = df[col].tolist()
        if col == 'addr':
            vals = [addr_words(v, c) for v, c in zip(vals, df.country.tolist())]
        for s in vals:
            c.update(set(s.split()))
            n += 1
    return {t: math.log((n + 1) / (v + 1)) for t, v in c.items()}


COLS = ['name', 'core', 'dom', 'alias', 'addr', 'house', 'postal', 'nums']


def id_codes(ids):
    """'S2-12345' -> 2*10^10 + 12345 (int64), vectorized."""
    s = pd.Series(ids)
    return (s.str[1].astype(np.int64) * 10_000_000_000 + s.str[3:].astype(np.int64)).values


class Table:
    """Id-indexed record table; materializes Python strings only for requested rows."""

    def __init__(self, df):
        self.df = df.reset_index(drop=True)
        self.index = pd.Index(id_codes(self.df.id))

    def take(self, ids, cols, prefix):
        codes = ids if np.issubdtype(np.asarray(ids).dtype, np.integer) else id_codes(ids)
        pos = self.index.get_indexer(codes)
        if (pos < 0).any():
            raise KeyError(f'{(pos < 0).sum()} ids not found')
        out = {prefix + c: np.asarray(self.df[c].take(pos).tolist(), dtype=object) for c in cols}
        if 'bnorm' in self.df:
            out[prefix + 'bnorm'] = np.asarray(self.df['bnorm'].take(pos).tolist(), dtype=np.float32)
        return out


def attach(cands, s1, tg):
    """Join normalized columns for both sides onto candidate pairs (by id).
    s1/tg may be DataFrames or Table instances."""
    s1 = s1 if isinstance(s1, Table) else Table(s1)
    tg = tg if isinstance(tg, Table) else Table(tg)
    s_ids = np.asarray(cands.s1.tolist(), dtype=object)
    t_ids = np.asarray(cands.tid.tolist(), dtype=object)
    d = {'s1': s_ids, 'tid': t_ids, 'bscore': cands.bscore.values, 'brank': cands.brank.values}
    d.update(s1.take(s_ids, COLS + ['country'], 's_'))
    d.update(tg.take(t_ids, COLS, 't_'))
    P = pd.DataFrame(d)
    P['is_s2'] = np.fromiter((t[1] == '2' for t in t_ids), dtype=np.int8, count=len(t_ids))
    return P
