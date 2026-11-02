#!/bin/bash
# One-time environment setup: .venv with the language-model extras + public lexicons.
# Submit from the repo root:  mkdir -p slurm/logs && sbatch -p <partition> slurm/setup_env.sh
# Optional: PYTHON_MODULE=<module to load> (else the python3 on PATH is used).
#SBATCH --job-name=linguistic_setup
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=slurm/logs/%x_%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"                       # not dirname "$0": sbatch runs a spooled copy

if [ -n "${PYTHON_MODULE:-}" ]; then module load "$PYTHON_MODULE"; fi
: "${LINGUISTIC_RESOURCES:?export LINGUISTIC_RESOURCES=<shared folder for lexicons/models> before sbatch}"

python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -e ".[lm,dev]"

.venv/bin/python -c "import numpy, torch, transformers, blick, linguistic; \
print('torch', torch.__version__, 'cuda', torch.version.cuda, '| transformers', transformers.__version__)"
.venv/bin/python -c "import nltk; nltk.download('cmudict', quiet=True)"
.venv/bin/linguistic fetch subtlex_us
.venv/bin/linguistic list
