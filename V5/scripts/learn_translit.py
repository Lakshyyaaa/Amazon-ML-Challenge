"""Learn a native-script word -> Latin word dictionary from aligned training pairs.

Uses only the provided training data: for every ground-truth (S1, target) pair where the
target contains non-Latin words, align words position-wise (names, and address components
split on commas) when word counts agree, then keep the majority Latin spelling.
"""
import pickle
import re
import unicodedata
from collections import Counter, defaultdict

import pandas as pd

from common import DATA, TRANSLIT_PATH, has_nonlatin

WORD = re.compile(r'\S+')


def words(s):
    return [unicodedata.normalize('NFC', w.strip('.,;:()[]{}')) for w in WORD.findall(s)]


def main():
    gt = pd.read_csv(f'{DATA}/train/train_ground_truth.tsv', sep='\t', dtype=str).dropna()
    gt = gt.assign(t=gt.matched_entity_ids.str.split(',')).explode('t')[['source1_entity_id', 't']]
    s1 = pd.read_csv(f'{DATA}/train/train_source1.tsv', sep='\t', dtype=str).set_index('entity_id')
    s1 = s1[s1.country == 'India']
    tg = []
    for f in ('train_source2', 'train_source3'):
        d = pd.read_csv(f'{DATA}/train/{f}.tsv', sep='\t', dtype=str)
        d = d[d.country == 'India']
        m = d.business_name.fillna('').map(has_nonlatin) | d.business_address.fillna('').map(has_nonlatin)
        tg.append(d[m])
    tg = pd.concat(tg).set_index('entity_id')
    gt = gt[gt.t.isin(tg.index) & gt.source1_entity_id.isin(s1.index)]
    print('aligned pairs with native script:', len(gt))

    votes = defaultdict(Counter)
    sn = s1.business_name.reindex(gt.source1_entity_id).fillna('').values
    sa = s1.business_address.reindex(gt.source1_entity_id).fillna('').values
    tn = tg.business_name.reindex(gt.t).fillna('').values
    ta = tg.business_address.reindex(gt.t).fillna('').values
    for a, b, c, d in zip(sn, tn, sa, ta):
        pairs = [(a, b)]
        ca, cb = c.split(','), d.split(',')
        # the state is usually the last component in both sources
        if has_nonlatin(cb[-1]):
            for _ in range(3):
                pairs.append((ca[-1], cb[-1]))
        # address components are reordered between sources: align native components by
        # matching word-count against every Latin component with the same length (voting fixes noise)
        for x in cb:
            if has_nonlatin(x):
                wx = words(x)
                for y in ca:
                    if len(words(y)) == len(wx):
                        pairs.append((y, x))
        for lat, nat in pairs:
            if not has_nonlatin(nat):
                continue
            wl, wn = words(lat), words(nat)
            if len(wn) == 1 and len(wl) > 1 and not has_nonlatin(lat):
                votes[wn[0]][' '.join(wl).lower()] += 1
                continue
            if len(wl) != len(wn):
                continue
            for l, n in zip(wl, wn):
                if has_nonlatin(n) and not has_nonlatin(l):
                    votes[n][l.lower()] += 1
    d = {}
    for n, c in votes.items():
        (best, k), tot = c.most_common(1)[0], sum(c.values())
        if k >= 2 and k / tot >= 0.4:
            d[n] = best
    with open(TRANSLIT_PATH, 'wb') as f:
        pickle.dump(d, f)
    print('dictionary size', len(d))
    for k in list(d)[:40]:
        print(' ', k, '->', d[k])


if __name__ == '__main__':
    main()
