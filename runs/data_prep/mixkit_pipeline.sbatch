#!/bin/bash
#SBATCH -A su007-hw-gpu
#SBATCH -p gpu
#SBATCH --gres=gpu:ampere_a100:2
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --mem-per-cpu=3850
#SBATCH -t 12:00:00
#SBATCH -J mixkit_prep
#SBATCH -o runs/data_prep/slurm_%x_%j.log
# Full Mixkit pipeline (spec §2.3): prep clips -> Qwen2.5-VL-7B captions (prompt v2, 2 GPUs) -> WDS for E2 and E2s.
# All steps are resumable; rerunning the job continues where it stopped.
set -uo pipefail
cd /gpfs/home/j/jiayang_gu/workspace/data-forcing-distillation
export HF_HOME=$PWD/FASTGEN_OUTPUT/.cache
PY=.venv-vlm/bin/python
PV=${PV:-v2}
nvidia-smi --query-gpu=name,memory.total --format=csv
date; echo "== prep"
$PY scripts/data_prep/prep_mixkit.py --src data/mixkit_src --out data/mixkit_prep --workers 30 || exit 1
$PY - <<'PYEOF'
import json
ok=[json.loads(l) for l in open("data/mixkit_prep/meta.jsonl")]
ok=[r["out"] for r in ok if r["status"]=="ok"]
open("data/mixkit_prep/caption_list.txt","w").write("\n".join(sorted(ok))+"\n"); print(len(ok),"clips to caption")
PYEOF
date; echo "== caption"
for g in 0 1; do
  CUDA_VISIBLE_DEVICES=$g $PY scripts/data_prep/caption_qwen.py --list data/mixkit_prep/caption_list.txt \
    --out data/mixkit_prep/captions_${PV}_s$g.jsonl --shard $g --nshards 2 --prompt_version $PV --batch 8 \
    > runs/data_prep/caption_mixkit_s$g.log 2>&1 &
done
wait
cat data/mixkit_prep/captions_${PV}_s0.jsonl data/mixkit_prep/captions_${PV}_s1.jsonl > data/mixkit_prep/captions_${PV}.jsonl
wc -l data/mixkit_prep/captions_${PV}.jsonl
date; echo "== build wds"
CUDA_VISIBLE_DEVICES=0 $PY scripts/data_prep/build_wds.py --captions data/mixkit_prep/captions_${PV}.jsonl \
  --meta data/mixkit_prep/meta.jsonl --out data --suffix "${SUFFIX:-}" || exit 1
date; echo PIPELINE_DONE
