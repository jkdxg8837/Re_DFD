#!/bin/bash
# Download the ViPE filtered set, `long` subfolder only (spec §2.3), low-memory mode for the Sulis login node.
set -e
cd /gpfs/home/j/jiayang_gu/workspace/data-forcing-distillation
export PATH=$PWD/.venv/bin:$PATH HF_HUB_ENABLE_HF_TRANSFER=0 HF_HUB_DISABLE_XET=1
hf download --repo-type dataset --max-workers 1 csusupergear/vipe_fliter_complete \
  --include "long/*" --local-dir data/vipe_fliter_complete
echo DONE
