#!/bin/bash
# k-sweep experiment: gold standard = original model at expansion width 50 (top-50);
# test the simplify model at expansion width k = 3..10. Everything else fixed
# (time limit 8 s, <=100 expansions, depth 5, 60 s hard cap), so only k varies.
# Per-molecule bb_coverage, expansions and wall-clock time are recorded for every
# config on all 1000 ChEMBL targets, for later analysis of the
# efficiency / scoring-accuracy trade-off vs k.
source ~/miniconda3/etc/profile.d/conda.sh && conda activate template-gnn
export PYTHONPATH=~/synomega_pkg/src
cd ~/Projects/retrosyn/benchmarks/chembl_rerun

CMP=~/Projects/retrosyn/benchmarks/simplify_eval/scripts/compare_simplify_vs_original.py
STOCK=~/synomega_assets/zinc_stock_keys.txt.gz
BASE="--time-limit 8 --max-expansions 100 --hard-timeout 60"
mkdir -p results/ksweep/shards
split -n l/8 -d data/targets.smi results/ksweep/shards/t_

run_config () {   # $1=run_dir  $2=width  $3=outname
  for i in 0 1 2 3 4 5 6 7; do
    gpu=$((i % 2))
    python $CMP --run-dir "$1" --stock $STOCK \
      --smiles results/ksweep/shards/t_0$i --out results/ksweep/shards/${3}_0$i.csv \
      $BASE --expansion-width "$2" --device cuda:$gpu \
      > results/ksweep/shards/${3}_0$i.log 2>&1 &
  done
  wait
  head -1 results/ksweep/shards/${3}_00.csv > results/ksweep/${3}.csv
  for i in 0 1 2 3 4 5 6 7; do tail -n +2 results/ksweep/shards/${3}_0$i.csv >> results/ksweep/${3}.csv; done
  echo "done ${3}: $(($(wc -l < results/ksweep/${3}.csv)-1)) rows"
}

run_config ~/synomega_pkg/run_r20 50 gold_original_k50
for k in 3 4 5 6 7 8 9 10; do
  run_config ~/synomega_pkg/run_simplify "$k" "simplify_k$(printf %02d "$k")"
done
echo "KSWEEP_DONE"
