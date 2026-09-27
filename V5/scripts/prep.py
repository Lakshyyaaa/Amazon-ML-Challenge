"""Normalize every source file once and cache as parquet: V5/cache/{split}_s{1,2,3}.parquet"""
import os
import sys
import time
from multiprocessing import Pool

import pandas as pd

from common import CACHE, DATA, addr_fields, name_fields


def _norm(df):
    rows = []
    for eid, n, a, c in zip(df.entity_id, df.business_name, df.business_address, df.country):
        c = str(c).strip()
        nc, core, dom, alias = name_fields(n)
        an, house, postal, nums = addr_fields(a, c)
        rows.append((eid, c, nc, core, dom, alias, an, house, postal, nums))
    return pd.DataFrame(rows, columns=['id', 'country', 'name', 'core', 'dom', 'alias',
                                       'addr', 'house', 'postal', 'nums'])


def run(split, src):
    out = f'{CACHE}/{split}_s{src}.parquet'
    if os.path.exists(out):
        return
    t = time.time()
    df = pd.read_csv(f'{DATA}/{split}/{split}_source{src}.tsv', sep='\t', dtype=str, keep_default_na=False)
    chunks = [df.iloc[i:i + 50000] for i in range(0, len(df), 50000)]
    with Pool(10) as p:
        res = pd.concat(p.map(_norm, chunks), ignore_index=True)
    res.to_parquet(out, index=False)
    print(f'{split} s{src}: {len(res):,} rows in {time.time() - t:.0f}s', flush=True)


if __name__ == '__main__':
    splits = sys.argv[1:] or ['train', 'test']
    for sp in splits:
        for s in (1, 2, 3):
            run(sp, s)
