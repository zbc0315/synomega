#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate template-gnn
cd ~/synomega_pkg
export PYTHONPATH=src
for s in 00 01 02 03 04 05 06 07; do
  CUDA_VISIBLE_DEVICES=1 python /tmp/score_synomega.py \
    --run-dir /home/zbc/synomega_pkg/run_r20 \
    --stock /home/zbc/synomega_assets/zinc_stock_keys.txt.gz \
    --smiles /tmp/shard_$s --out /tmp/syn_shard_$s.csv \
    --max-steps 5 --time-limit 12 --max-expansions 150 --device cuda:0 \
    > /tmp/syn_shard_$s.log 2>&1 &
done
wait
echo "ALL SHARDS DONE"
head -q -n1 /tmp/syn_shard_00.csv > /tmp/syn_targets.csv
for s in 00 01 02 03 04 05 06 07; do tail -n +2 /tmp/syn_shard_$s.csv >> /tmp/syn_targets.csv; done
echo "merged $(($(wc -l < /tmp/syn_targets.csv)-1)) rows -> /tmp/syn_targets.csv"
