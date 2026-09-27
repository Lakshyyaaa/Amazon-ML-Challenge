#!/bin/zsh
# Block + featurize every train fold separately (bounded memory). Usage: ./build_all_folds.sh 0 1 2 ...

cd "$(dirname "$0")"
for f in "$@"; do
  if [[ ! -f ../cache/cands_train_f$f.parquet ]]; then
    /opt/homebrew/bin/python3.11 run_blocking.py --split train --folds $f --tag f$f --k 50 2>&1 | grep -E "recall@50|saved|Error"
  fi
  if [[ ! -f ../cache/feat_train_f$f.parquet ]]; then
    /opt/homebrew/bin/python3.11 build_features.py --split train --tag f$f 2>&1 | grep -E "saved|Error"
  fi
done
