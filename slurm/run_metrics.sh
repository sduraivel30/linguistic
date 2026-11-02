#!/bin/bash
# Run the pipeline as a batch job. Submit from the repo root:
#     sbatch -p <partition> [--gres=gpu:1] slurm/run_metrics.sh <stimuli.csv> <out dir> [metrics ...]
# Metrics are passed to `linguistic run -m` (default: all).
#SBATCH --job-name=linguistic_run
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
