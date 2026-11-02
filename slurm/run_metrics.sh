#!/bin/bash
# Run the pipeline on a SLURM cluster. Submit from the repo root:
#     sbatch slurm/run_metrics.sh <stimuli.csv> <out dir> [metric selectors ...]
# e.g. sbatch --gres=gpu:1 slurm/run_metrics.sh data/my_stimuli.csv out/my_run lm
# Selectors are passed to `linguistic run -m` (default: all). Add --gres=gpu:1 for
# language models; small stimulus sets also run on CPU.
#SBATCH --job-name=linguistic_run
#SBATCH --partition=pi_evelina9,ou_bcs_normal
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=slurm/logs/%x_%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
: "${LINGUISTIC_RESOURCES:?export LINGUISTIC_RESOURCES=<shared folder for lexicons/models> before sbatch}"

input=$1; out=$2; shift 2
sel=(); for m in "${@:-all}"; do sel+=(-m "$m"); done

nvidia-smi --query-gpu=name,memory.total --format=csv 2>/dev/null || echo "no GPU: running on CPU"
.venv/bin/linguistic run -i "$input" -o "$out" "${sel[@]}"
