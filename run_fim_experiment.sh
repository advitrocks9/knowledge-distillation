#!/usr/bin/env bash
# Mellum-as-teacher seq-KD experiment, end-to-end.
# Run from the project dir on a CUDA host (4090 or Modal A10G).

set -euo pipefail

PY=~/kd-mellum-qwen/.venv/bin/python
mkdir -p logs results data/fim

echo "=== fim data ($(date)) ==="
$PY -u fim_data.py --n-per-kind 400 --skip 10000 --seed 0 --out data/fim/train_examples.jsonl 2>&1 | tail -10
$PY -u fim_data.py --n-per-kind 100 --skip 18000 --seed 1 --out data/fim/eval_examples.jsonl 2>&1 | tail -10

echo "=== mellum generation ($(date)) ==="
$PY -u fim_generate.py --in-jsonl data/fim/train_examples.jsonl --out-jsonl data/fim/mellum_completions.jsonl --max-new 200 --greedy 2>&1 | tail -20

for src in gold mellum mix; do
  echo "=== fim train ${src} ($(date)) ==="
  $PY -u fim_train.py --middle ${src} --steps 1500 --lr 2e-5 --eval-every 300 2>&1 | tail -20
done

echo "=== fim eval ($(date)) ==="
$PY -u fim_eval.py --limit 300 2>&1 | tail -30

echo "=== humaneval infilling ($(date)) ==="
$PY -u humaneval_infilling.py --n-problems 164 --max-new 256 2>&1 | tail -30

echo "=== done ($(date)) ==="
