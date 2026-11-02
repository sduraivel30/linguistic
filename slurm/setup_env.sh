#!/bin/bash
# One-time environment setup on a SLURM cluster (tested on MIT ORCD).
# Builds .venv in the repo with the language-model extras (torch, transformers) and
# fetches the lexicons that have a public URL. Submit from the repo root:
#     mkdir -p slurm/logs && sbatch slurm/setup_env.sh
#SBATCH --job-name=linguistic_setup
#SBATCH --partition=pi_evelina9,ou_bcs_normal
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=slurm/logs/%x_%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"                       # not dirname "$0": sbatch runs a spooled copy

module load "${MINIFORGE_MODULE:-miniforge/26.7.2-0}"
: "${LINGUISTIC_RESOURCES:?export LINGUISTIC_RESOURCES=<shared folder for lexicons/models> before sbatch}"

python -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -e ".[lm,dev]"

.venv/bin/python -c "import numpy, torch, transformers, blick, linguistic; \
print('torch', torch.__version__, 'cuda', torch.version.cuda, '| transformers', transformers.__version__)"
.venv/bin/python -c "import nltk; nltk.download('cmudict', quiet=True)"
.venv/bin/linguistic fetch subtlex_us
.venv/bin/linguistic list
