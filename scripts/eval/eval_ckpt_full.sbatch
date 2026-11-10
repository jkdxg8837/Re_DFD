#!/bin/bash
#SBATCH -A mlsys
#SBATCH -p mlsys-ada
#SBATCH --gres=gpu:2
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 24
#SBATCH --mem-per-cpu=5000
#SBATCH -t 16:00:00
#SBATCH -J dfd_eval_full
#SBATCH -o runs/eval/slurm_%x_%j.log
# Full VBench eval of one checkpoint (task2.md B2.1: Total / Quality / Semantic Score). Opt-in; eval_ckpt.sbatch is
# unchanged and still the default light eval.
#   1. scripts/eval/eval_ckpt.sbatch as is: 160 quality prompts x 5 seeds (runs/eval/<tag>/videos), 7 quality dims,
#      Quality Score, saturation / diversity extras.
#   2. the 280 prompts of runs/eval/vbench_subset_full/prompts_new.txt x the same 5 seeds -> runs/eval/<tag>/raw_sem;
#      runs/eval/<tag>/videos_full = all 440 prompts (quality videos reused for the shared scene /
#      overall_consistency prompts, so every video exists once).
#   3. VBench standard mode, 9 semantic dims (runs/eval/vbench_subset_full/subset_full_info.json, with the official
#      auxiliary_info), split over the 2 GPUs -> runs/eval/<tag>/vbench_sem/.
#   4. scripts/eval/vbench_total_score.py -> runs/eval/<tag>/total_score.json.
# Usage: sbatch --export=ALL,CKPT=<ckpt_prefix>,TAG=<tag>,T=<latent_T> scripts/eval/eval_ckpt_full.sbatch
set -euo pipefail
cd /dcs/large/u5649209/workspace/data-forcing-distillation
: "${CKPT:?}" "${TAG:?}"
T=${T:-21}
export VBENCH_CACHE_DIR=$PWD/data/vbench_cache HF_HOME=/dcs/large/u5649209/cache
OUT=runs/eval/$TAG
SUB=runs/eval/vbench_subset_full
t0=$(date +%s)
bash scripts/eval/eval_ckpt.sbatch
t1=$(date +%s)
n_new=$(grep -c . $SUB/prompts_new.txt)
have=$(ls $OUT/raw_sem/seed*/*/ 2>/dev/null | grep -c '\.mp4$' || true)
if [ "$have" -eq $((n_new * 5)) ]; then
  echo "all $have semantic videos present for $TAG, skipping generation"
else
  PROMPTS=$SUB/prompts_new.txt RAW=raw_sem SUFFIX=_sem LINK=0 scripts/eval/gen_vbench_subset.sh "$CKPT" "$TAG" "$T"
fi
python3 scripts/eval/link_vbench_names.py --prompts runs/eval/vbench_subset/prompts.txt --raw $OUT/raw --out $OUT/videos_full
python3 scripts/eval/link_vbench_names.py --prompts $SUB/prompts_new.txt --raw $OUT/raw_sem --out $OUT/videos_full
t2=$(date +%s)
SEM=(object_class multiple_objects human_action color spatial_relationship scene appearance_style temporal_style overall_consistency)
mkdir -p $OUT/vbench_sem $OUT/logs
CUDA_VISIBLE_DEVICES=0 .venv-vbench/bin/python scripts/eval/run_vbench.py --videos $OUT/videos_full --name ${TAG}_semA \
  --out $OUT/vbench_sem --full_info $SUB/subset_full_info.json --dims ${SEM[@]:0:5} > $OUT/logs/vbench_semA.log 2>&1 &
PA=$!
CUDA_VISIBLE_DEVICES=1 .venv-vbench/bin/python scripts/eval/run_vbench.py --videos $OUT/videos_full --name ${TAG}_semB \
  --out $OUT/vbench_sem --full_info $SUB/subset_full_info.json --dims ${SEM[@]:5} > $OUT/logs/vbench_semB.log 2>&1 &
PB=$!
wait $PA; wait $PB
python3 - $OUT/vbench_sem ${TAG} <<'PY'
import json, os, sys
d, tag = sys.argv[1:3]
m = {}
for part in ("semA", "semB"):
    m.update(json.load(open(os.path.join(d, f"{tag}_{part}_eval_results.json"))))
json.dump(m, open(os.path.join(d, f"{tag}_sem_eval_results.json"), "w"))
print({k: v[0] for k, v in m.items()})
PY
python3 scripts/eval/vbench_total_score.py $OUT/vbench/${TAG}_eval_results.json \
  $OUT/vbench_sem/${TAG}_sem_eval_results.json $OUT/total_score.json
t3=$(date +%s)
echo "timing_s quality_part=$((t1 - t0)) semantic_gen=$((t2 - t1)) semantic_vbench=$((t3 - t2)) total=$((t3 - t0))" | tee $OUT/timing_full.txt
du -sh --apparent-size $OUT | tee -a $OUT/timing_full.txt
echo EVAL_FULL_DONE
