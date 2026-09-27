"""Train a stage-1 pair model from per-fold feature shards.

python stage1_train.py --folds 1,2 --es 19 --out s1A
"""
import argparse
import time

import lightgbm as lgb
import pandas as pd
from sklearn.metrics import roc_auc_score

from common import CACHE, V5

DROP = {'s1', 'tid', 'label', 's_legal_mask', 't_legal_mask'}
PARAMS = dict(objective='binary', learning_rate=0.08, num_leaves=127, min_data_in_leaf=200,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
              num_threads=10, verbose=-1, max_bin=255)


def load_folds(folds):
    return pd.concat([pd.read_parquet(f'{CACHE}/feat_train_f{f}.parquet') for f in folds], ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--folds', required=True)
    ap.add_argument('--es', type=int, default=19)
    ap.add_argument('--rounds', type=int, default=3000)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    t0 = time.time()
    tr = load_folds([int(x) for x in a.folds.split(',')])
    es = pd.read_parquet(f'{CACHE}/feat_train_f{a.es}.parquet')
    feats = [c for c in tr.columns if c not in DROP]
    print(f'train {len(tr):,} rows, {len(feats)} features')
    dtr = lgb.Dataset(tr[feats], tr.label)
    des = lgb.Dataset(es[feats], es.label, reference=dtr)
    m = lgb.train(PARAMS, dtr, a.rounds, valid_sets=[des],
                  callbacks=[lgb.log_evaluation(500), lgb.early_stopping(100)])
    m.save_model(f'{V5}/models/{a.out}.txt')
    p = m.predict(es[feats], num_threads=10)
    print(f'{a.out}: es AUC {roc_auc_score(es.label, p):.5f}, {m.best_iteration} rounds, {time.time() - t0:.0f}s')
    imp = pd.Series(m.feature_importance('gain'), feats).sort_values(ascending=False)
    print((imp / imp.sum()).round(4).head(15).to_string())


if __name__ == '__main__':
    main()
