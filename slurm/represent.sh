#!/bin/bash
# Layer-wise activations and embeddings as a batch job. Submit from the repo root:
#     sbatch -p <partition> --gres=gpu:1 slurm/represent.sh <stimuli.csv> <out dir> <model> [model ...]
# Extra options (e.g. --context full --dtype float16) go in REPRESENT_ARGS.
#SBATCH --job-name=linguistic_represent
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=slurm/logs/%x_%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
: "${LINGUISTIC_RESOURCES:?export LINGUISTIC_RESOURCES=<shared folder for lexicons/models> before sbatch}"

input=$1; out=$2; shift 2
sel=(); for m in "$@"; do sel+=(-m "$m"); done

nvidia-smi --query-gpu=name,memory.total --format=csv 2>/dev/null || echo "no GPU: running on CPU"
.venv/bin/linguistic represent -i "$input" -o "$out" "${sel[@]}" ${REPRESENT_ARGS:-}
