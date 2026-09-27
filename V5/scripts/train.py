"""Train the stage-1 pair model on feat_train_tr12 and evaluate macro F0.5 on feat_train_val."""
import argparse
import json
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from common import CACHE, DATA, V5
from selection import load_gt, macro_f05, select_expected_f, select_threshold

DROP = {'s1', 'tid', 'label'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train', default='tr12')
    ap.add_argument('--val', default='val')
    ap.add_argument('--rounds', type=int, default=1500)
    ap.add_argument('--out', default='stage1')
    a = ap.parse_args()
    t0 = time.time()
    tr = pd.read_parquet(f'{CACHE}/feat_train_{a.train}.parquet')
    va = pd.read_parquet(f'{CACHE}/feat_train_{a.val}.parquet')
    feats = [c for c in tr.columns if c not in DROP]
    print(f'train {len(tr):,} (pos {tr.label.mean():.3f}), val {len(va):,}, {len(feats)} features')
    params = dict(objective='binary', learning_rate=0.08, num_leaves=127, min_data_in_leaf=200,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                  num_threads=10, verbose=-1, max_bin=255)
    dtr = lgb.Dataset(tr[feats], tr.label, free_raw_data=True)
    dva = lgb.Dataset(va[feats], va.label, reference=dtr)
    m = lgb.train(params, dtr, a.rounds, valid_sets=[dva],
                  callbacks=[lgb.log_evaluation(250), lgb.early_stopping(100)])
    m.save_model(f'{V5}/models/{a.out}.txt')
    va['p'] = m.predict(va[feats], num_threads=10)
    print(f'val AUC {roc_auc_score(va.label, va.p):.5f}  ({time.time() - t0:.0f}s)')
    imp = pd.Series(m.feature_importance('gain'), feats).sort_values(ascending=False)
    print((imp / imp.sum()).round(4).head(25).to_string())

    s1_ids = set(pd.read_parquet(f'{CACHE}/cands_train_{a.val}.parquet', columns=['s1']).s1)
    gt = load_gt(f'{DATA}/train/train_ground_truth.tsv', s1_ids)
    # S1 with zero candidates are also in gt (scored as empty)
    va[['s1', 'tid', 'p', 'label']].to_parquet(f'{CACHE}/pred_{a.out}_{a.val}.parquet', index=False)
    print('\nthreshold policy:')
    for tau in (0.3, 0.4, 0.5, 0.6, 0.7):
        print(f'  tau={tau}: F0.5={macro_f05(select_threshold(va, tau=tau), gt):.4f}')
    print('expected-F policy:')
    for mm in (0.0, 0.05, 0.1):
        print(f'  miss_mass={mm}: F0.5={macro_f05(select_expected_f(va, miss_mass=mm), gt):.4f}')


if __name__ == '__main__':
    main()
