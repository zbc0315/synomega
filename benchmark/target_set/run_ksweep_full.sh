#!/bin/bash
# k-sweep for the FULL (unconstrained) model at expansion width k = 3..10, same
# fixed budget as the simplify sweep (8 s / <=100 expansions / depth 5 / 60 s cap).
# Complements run_ksweep_chembl.sh (gold = full@50, simplify k=3..10): together they
# show how full and simplify each approach the full@50 gold as k grows, and their
# efficiency / scoring-accuracy trade-offs. Per-molecule bb_coverage/expansions/time
# recorded on all 1000 ChEMBL targets.
source ~/miniconda3/etc/profile.d/conda.sh && conda activate template-gnn
export PYTHONPATH=~/synomega_pkg/src
cd ~/Projects/retrosyn/benchmarks/chembl_rerun

CMP=~/Projects/retrosyn/benchmarks/simplify_eval/scripts/compare_simplify_vs_full.py
STOCK=~/synomega_assets/zinc_stock_keys.txt.gz
BASE="--time-limit 8 --max-expansions 100 --hard-timeout 60"
mkdir -p results/ksweep/shards
split -n l/8 -d data/targets.smi results/ksweep/shards/tf_

run_config () {   # $1=run_dir  $2=width  $3=outname
  for i in 0 1 2 3 4 5 6 7; do
    gpu=$((i % 2))
    python $CMP --run-dir "$1" --stock $STOCK \
      --smiles results/ksweep/shards/tf_0$i --out results/ksweep/shards/${3}_0$i.csv \
      $BASE --expansion-width "$2" --device cuda:$gpu \
      > results/ksweep/shards/${3}_0$i.log 2>&1 &
  done
  wait
  head -1 results/ksweep/shards/${3}_00.csv > results/ksweep/${3}.csv
  for i in 0 1 2 3 4 5 6 7; do tail -n +2 results/ksweep/shards/${3}_0$i.csv >> results/ksweep/${3}.csv; done
  echo "done ${3}: $(($(wc -l < results/ksweep/${3}.csv)-1)) rows"
}

for k in 3 4 5 6 7 8 9 10; do
  run_config ~/synomega_pkg/run_r20 "$k" "full_k$(printf %02d "$k")"
done
echo "FULLSWEEP_DONE"
