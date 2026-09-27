"""Entity-level selection policies and the exact macro F0.5 scorer."""
import numpy as np
import pandas as pd

B2 = 0.25  # beta^2 for F0.5


def macro_f05(pred, gt):
    """pred, gt: dict s1 -> set(ids). Every key of gt is scored (missing pred = empty)."""
    tot = 0.0
    for s, g in gt.items():
        p = pred.get(s, set())
        if not g:
            tot += 1.0 if not p else 0.0
            continue
        if not p:
            continue
        tp = len(p & g)
        if tp == 0:
            continue
        prec, rec = tp / len(p), tp / len(g)
        tot += (1 + B2) * prec * rec / (B2 * prec + rec)
    return tot / len(gt)


def select_expected_f(df, prob='p', miss_mass=0.0, min_p=0.0, kmax=12):
    """Pick, per S1, the top-k (by prob) that maximizes plug-in expected F0.5.

    df has columns s1, tid, prob. miss_mass = expected number of true matches the blocker
    missed per S1 (adds to the FN term). Returns dict s1 -> set(tid).
    """
    out = {}
    d = df[['s1', 'tid', prob]].sort_values(['s1', prob], ascending=[True, False])
    s1 = d.s1.values
    tid = d.tid.values
    p = d[prob].values.astype(np.float64)
    starts = np.flatnonzero(np.r_[True, s1[1:] != s1[:-1]])
    ends = np.r_[starts[1:], len(s1)]
    for a, b in zip(starts, ends):
        pp = p[a:b]
        total = pp.sum() + miss_mass
        p_none = np.exp(np.log1p(-np.clip(pp, 0, 1 - 1e-9)).sum())
        best_k, best_v = 0, p_none
        tp = 0.0
        for k in range(1, min(kmax, b - a) + 1):
            if pp[k - 1] < min_p:
                break
            tp += pp[k - 1]
            v = (1 + B2) * tp / ((1 + B2) * tp + B2 * (total - tp) + (k - tp))
            if v > best_v:
                best_v, best_k = v, k
        out[s1[a]] = set(tid[a:a + best_k])
    return out


def select_threshold(df, prob='p', tau=0.5):
    d = df[df[prob] >= tau]
    return d.groupby('s1').tid.agg(set).to_dict()


def load_gt(path, s1_ids):
    gt = pd.read_csv(path, sep='\t', dtype=str, keep_default_na=False)
    gt = gt[gt.source1_entity_id.isin(s1_ids)]
    return {s: set(m.split(',')) if m else set() for s, m in zip(gt.source1_entity_id, gt.matched_entity_ids)}
