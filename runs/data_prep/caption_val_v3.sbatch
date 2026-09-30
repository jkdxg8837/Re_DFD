#!/bin/bash
#SBATCH -A su007-hw-gpu
#SBATCH -p gpu-devel
#SBATCH --gres=gpu:ampere_a100:1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem-per-cpu=3850
#SBATCH -t 01:00:00
#SBATCH -J cap_val3
#SBATCH -o runs/data_prep/slurm_%x_%j.log
# Prompt v3 validation: 50 ViPE videos (vs original captions) + 100 random Mixkit clips (caption-vs-title check,
# compared with the v2 captions of the same 100 clips).
cd /gpfs/home/j/jiayang_gu/workspace/data-forcing-distillation
export HF_HOME=$PWD/FASTGEN_OUTPUT/.cache
PY=.venv-vlm/bin/python
$PY scripts/data_prep/caption_qwen.py --list data/caption_val/vipe_list.txt --out runs/data_prep/caption_val_v3.jsonl --prompt_version v3
$PY scripts/data_prep/compare_captions.py runs/data_prep/caption_val_v3.jsonl --out runs/data_prep/caption_val_v3.md
$PY scripts/data_prep/compare_captions.py runs/data_prep/caption_val_v2.jsonl --out runs/data_prep/caption_val_v2.md
$PY scripts/data_prep/caption_qwen.py --list data/caption_val/mixkit100_list.txt --out runs/data_prep/caption_mixkit100_v3.jsonl --prompt_version v3
$PY - <<'PYEOF'
import json
keep={l.strip() for l in open("data/caption_val/mixkit100_list.txt") if l.strip()}
with open("runs/data_prep/caption_mixkit100_v2.jsonl","w") as f:
    for l in open("data/mixkit_prep/captions_v2.jsonl"):
        if json.loads(l)["path"] in keep: f.write(l)
PYEOF
for v in v2 v3; do
  $PY scripts/data_prep/check_caption_title.py runs/data_prep/caption_mixkit100_$v.jsonl --out runs/data_prep/caption_title_mixkit100_$v
done
echo VAL3_DONE
