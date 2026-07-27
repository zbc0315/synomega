#!/bin/bash
# Re-run SynOmega scoring on all 1000 ChEMBL targets, unified budget
# (top-10 / 8 s / 100 expansions / depth 5 / 60 s hard cap) matching the paper.
#   full model  (run_r20)      on all 1000 -> correlation data + efficiency "full" arm
#   simplify    (run_simplify)  on all 1000 -> efficiency "simplify" arm
# Both sharded 8-way across 2 GPUs. Run from benchmarks/chembl_rerun.
set -e
source ~/miniconda3/etc/profile.d/conda.sh && conda activate template-gnn
export PYTHONPATH=~/synomega_pkg/src
cd ~/Projects/retrosyn/benchmarks/chembl_rerun

CMP=~/Projects/retrosyn/benchmarks/simplify_eval/scripts/compare_simplify_vs_full.py
STOCK=~/synomega_assets/zinc_stock_keys.txt.gz
BUDGET="--time-limit 8 --max-expansions 100 --expansion-width 10 --hard-timeout 60"
mkdir -p results/shards

split -n l/8 -d data/targets.smi results/shards/full_shard_
split -n l/8 -d data/targets.smi results/shards/simp_shard_

for i in 0 1 2 3 4 5 6 7; do
  gpu=$((i % 2))
  python $CMP --run-dir ~/synomega_pkg/run_r20 --stock $STOCK \
    --smiles results/shards/full_shard_0$i --out results/shards/full_out_0$i.csv \
    $BUDGET --device cuda:$gpu > results/shards/full_0$i.log 2>&1 &
  python $CMP --run-dir ~/synomega_pkg/run_simplify --stock $STOCK \
    --smiles results/shards/simp_shard_0$i --out results/shards/simp_out_0$i.csv \
    $BUDGET --device cuda:$gpu > results/shards/simp_0$i.log 2>&1 &
done
wait

head -1 results/shards/full_out_00.csv > results/full_1000.csv
head -1 results/shards/simp_out_00.csv > results/simplify_1000.csv
for i in 0 1 2 3 4 5 6 7; do
  tail -n +2 results/shards/full_out_0$i.csv >> results/full_1000.csv
  tail -n +2 results/shards/simp_out_0$i.csv >> results/simplify_1000.csv
done
echo "SYNOMEGA_DONE full=$(($(wc -l < results/full_1000.csv)-1)) simplify=$(($(wc -l < results/simplify_1000.csv)-1))"
