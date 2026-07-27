#!/bin/bash
# AiZynthFinder on ALL 1000 ChEMBL targets, on GPU (onnxruntime CUDA provider), with
# search depth & width aligned to SynOmega (cutoff_number=10, max_transforms=5,
# iteration_limit=100) via aizynth_config_aligned.yml. USPTO expansion + filter,
# ZINC stock. Running on GPU matches SynOmega's hardware for a fair timing comparison.
source ~/miniconda3/etc/profile.d/conda.sh && conda activate aizynth
# expose the pip-installed CUDA/cuDNN libs so onnxruntime uses the CUDA provider
export LD_LIBRARY_PATH=$(python -c "import os,glob,nvidia; b=os.path.dirname(nvidia.__file__); print(':'.join(glob.glob(b+'/*/lib')))"):$LD_LIBRARY_PATH
cd ~/Projects/retrosyn/benchmarks/chembl_rerun
CFG=~/Projects/retrosyn/benchmarks/chembl_rerun/scripts/aizynth_config_aligned.yml
aizynthcli --config $CFG --smiles data/targets.smi \
  --policy uspto --stocks zinc \
  --output results/aizynth_1000.json.gz --nproc 8
echo "AIZYNTH_1000_GPU_DONE"
