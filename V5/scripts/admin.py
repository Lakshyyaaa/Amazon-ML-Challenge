"""Learn per-country administrative phrases (state / region / department) from the data itself.

A phrase is 'admin' if it is the LAST comma component of an address in >= 70% of the addresses
where it appears as a component, and occurs >= 200 times. Cities fail this (mostly mid-address).
Output: cache/admin_{split}.pkl = {country: [phrase, ...]} (normalized, longest first).
"""
import os
import pickle
import sys
from collections import Counter, defaultdict

import pandas as pd

from common import CACHE, DATA, addr_fields


def build(split):
    path = f'{CACHE}/admin_{split}.pkl'
    if os.path.exists(path):
        return pickle.load(open(path, 'rb'))
    last, anyc = defaultdict(Counter), defaultdict(Counter)
    for f in (1, 2, 3):
        d = pd.read_csv(f'{DATA}/{split}/{split}_source{f}.tsv', sep='\t', dtype=str, keep_default_na=False,
                        usecols=['business_address', 'country'])
        for a, c in zip(d.business_address.values, d.country.values):
            if not a:
                continue
            comps = [addr_fields(x, c)[0] for x in a.split(',')]
            comps = [x for x in comps if x and not any(ch.isdigit() for ch in x)]
            if not comps:
                continue
            last[c][comps[-1]] += 1
            for x in set(comps):
                anyc[c][x] += 1
    out = {}
    for c in last:
        ph = [p for p, n in last[c].items() if n >= 200 and n / anyc[c][p] >= 0.7]
        out[c] = sorted(ph, key=len, reverse=True)
    pickle.dump(out, open(path, 'wb'))
    return out


if __name__ == '__main__':
    r = build(sys.argv[1])
    for c, ph in r.items():
        print(c, len(ph), ph[:25])
