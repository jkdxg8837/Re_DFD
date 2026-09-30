#!/bin/bash
# Runs on the login node (no torch): wait for both inversion jobs, pack the E4 WDS, then submit probe -> SMOKE_E4.
cd /gpfs/home/j/jiayang_gu/workspace/data-forcing-distillation
while squeue -j 2223372,2223373 -h 2>/dev/null | grep -q .; do sleep 120; done
n=$(cat data/mixkit_inv/meta_s*.jsonl | wc -l); echo "inverted: $n"
[ "$n" -ge 1280 ] || { echo "INCOMPLETE inversion ($n/1280), not building"; exit 1; }
nice .venv/bin/python scripts/data_prep/build_inv_wds.py --list data/mixkit_inv/subset1280.txt --inv data/mixkit_inv --out data/mixkit_inv_wds_v3 || exit 1
P=$(sbatch --parsable runs/probe_frames_E4/probe.sbatch); S=$(sbatch --parsable --dependency=afterany:$P runs/sbatch/SMOKE_E4.sbatch)
echo "probe $P smoke $S"; echo AFTER_INVERT_DONE
