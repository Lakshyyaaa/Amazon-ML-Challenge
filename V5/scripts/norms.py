"""Per-record blocker self-score (sum of IDF of its own hashed features), per split.

Lets features use a normalized blocker similarity (dice = 2*shared / (self_q + self_t)),
which transfers across countries better than the raw IDF sum.
Output: cache/norms_{split}.parquet (id, bnorm)
"""
import sys

import numpy as np
import pandas as pd

from blocker import NBITS, _featurize_par
from common import CACHE
from run_blocking import load_targets


def main(split):
    s1 = pd.read_parquet(f'{CACHE}/{split}_s1.parquet')
    tg = load_targets(split)
    out = []
    for c in sorted(s1.country.unique()):
        t = tg[tg.country == c].reset_index(drop=True)
        q = s1[s1.country == c].reset_index(drop=True)
        T = _featurize_par(t)
        df = np.bincount(T.indices, minlength=1 << NBITS)
        idf = np.log((T.shape[0] + 1) / (df + 1)).astype(np.float32)
        idf[df > 2000] = 0
        idf[df == 0] = 0
        for frame, M in ((t, T), (q, _featurize_par(q))):
            M.data = idf[M.indices]
            out.append(pd.DataFrame({'id': frame.id.values, 'bnorm': np.asarray(M.sum(axis=1)).ravel()}))
        print(c, 'done', flush=True)
    pd.concat(out, ignore_index=True).to_parquet(f'{CACHE}/norms_{split}.parquet', index=False)


if __name__ == '__main__':
    main(sys.argv[1])
