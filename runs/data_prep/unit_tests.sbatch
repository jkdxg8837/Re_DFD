#!/bin/bash
#SBATCH -A su007-hw-gpu
#SBATCH -p gpu-devel
#SBATCH --gres=gpu:ampere_a100:1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem-per-cpu=3850
#SBATCH -t 00:30:00
#SBATCH -J dfd_tests
#SBATCH -o runs/data_prep/slurm_%x_%j.log
# Unit tests for the DFD flags (CPU-only jobs are not allowed on this cluster, so tests run in a GPU job).
cd /gpfs/home/j/jiayang_gu/workspace/data-forcing-distillation
source .venv/bin/activate
python -m pytest -q --tb=short tests/test_dfd_flags.py tests/test_full_save_iter.py tests/test_dmd2model.py 2>&1 | grep -v "| DEBUG\|| INFO\|| WARNING" | tail -150
echo TESTS_DONE
