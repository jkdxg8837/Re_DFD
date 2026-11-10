#!/bin/bash
# Launch one experiment run with the bookkeeping required by spec §0 rule 4.
#   runs/<EXP_ID>/cmd.sh           exact command
#   runs/<EXP_ID>/git_diff.patch   code changes vs. the original DFD commit (tracked diff + untracked files)
#   runs/<EXP_ID>/train.log        full log
#   runs/<EXP_ID>/config_dump.yaml effective config (copied from the FastGen save path after startup)
#   runs/<EXP_ID>/load_report.json per-submodule load report (scripts/train_dfd.py)
# Usage (inside a Slurm allocation, from the repo root): scripts/launch_run.sh <EXP_ID> <NGPU> <override> [...]
set -uo pipefail
EXP_ID=$1
NGPU=$2
shift 2
REPO=/dcs/large/u5649209/workspace/data-forcing-distillation
cd $REPO
source .venv/bin/activate
export WANDB_MODE=offline
BASE_COMMIT=de6416c
D=runs/$EXP_ID
mkdir -p $D
{
  echo "#!/bin/bash"
  echo "# $(date -Is) on $(hostname), SLURM_JOB_ID=${SLURM_JOB_ID:-none}"
  printf 'cd %q && source .venv/bin/activate && WANDB_MODE=offline torchrun --nproc_per_node=%s --standalone scripts/train_dfd.py --config fastgen/configs/experiments/WanT2V/config_dmd2.py -' "$REPO" "$NGPU"
  for o in "$@"; do printf ' %q' "$o"; done
  echo
} > $D/cmd.sh
{
  echo "# base commit: $BASE_COMMIT ($(git log -1 --format=%s $BASE_COMMIT))"
  echo "# HEAD: $(git rev-parse --short HEAD)"
  git diff $BASE_COMMIT
  for f in $(git ls-files --others --exclude-standard -- fastgen scripts tests train.py); do
    git diff --no-index /dev/null "$f"
  done
} > $D/git_diff.patch
echo "[launch_run] $EXP_ID: $(tail -1 $D/cmd.sh)"
torchrun --nproc_per_node=$NGPU --standalone scripts/train_dfd.py \
  --config fastgen/configs/experiments/WanT2V/config_dmd2.py - "$@" > $D/train.log 2>&1
rc=$?
CFG=$(ls -t FASTGEN_OUTPUT/*/*/"$(grep -oE 'log_config.name=[^ ]+' $D/cmd.sh | head -1 | cut -d= -f2)"/config.yaml 2>/dev/null | head -1)
[ -n "$CFG" ] && cp "$CFG" $D/config_dump.yaml && cp "$(dirname "$CFG")/load_report.json" $D/ 2>/dev/null
echo "[launch_run] $EXP_ID finished with exit code $rc (config: ${CFG:-not found})"
exit $rc
