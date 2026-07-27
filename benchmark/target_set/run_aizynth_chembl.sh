#!/bin/bash
# AiZynthFinder on ALL 1000 ChEMBL targets, with search depth & width aligned to
# SynOmega's budget (cutoff_number=10, max_transforms=5, iteration_limit=100) via
# aizynth_config_aligned.yml. USPTO expansion + USPTO filter + ZINC stock.
# (The 200-subset and default-settings runs are deprecated.)
source ~/miniconda3/etc/profile.d/conda.sh && conda activate aizynth
cd ~/Projects/retrosyn/benchmarks/chembl_rerun
CFG=~/Projects/retrosyn/benchmarks/chembl_rerun/scripts/aizynth_config_aligned.yml
aizynthcli --config $CFG --smiles data/targets.smi \
  --policy uspto --stocks zinc \
  --output results/aizynth_1000.json.gz --nproc 8
echo "AIZYNTH_1000_DONE"
