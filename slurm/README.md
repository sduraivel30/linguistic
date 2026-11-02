# Running on ORCD

Needed mainly for the language models. Run everything from the repo root, through sbatch.

1. Clone the repo and set a folder for models and lexicons (add the export to `~/.bashrc`):
   ```bash
   cd /orcd/data/evelina9/001/USERS/sdurai30/repos
   git clone https://github.com/sduraivel30/linguistic.git && cd linguistic
   export LINGUISTIC_RESOURCES=/orcd/data/evelina9/001/USERS/sdurai30/data/linguistic_resources
   mkdir -p $LINGUISTIC_RESOURCES slurm/logs
   ```
2. Build the environment (once):
   ```bash
   sbatch slurm/setup_env.sh
   ```
3. Download each model you need (once):
   ```bash
   srun -p pi_evelina9 -t 00:30:00 --mem 8G .venv/bin/linguistic fetch gpt2-xl
   ```
4. Copy your stimulus file to the server, e.g. into `data/`.
5. Run (drop `--gres` to run on CPU):
   ```bash
   sbatch --gres=gpu:1 slurm/run_metrics.sh data/my_stimuli.csv out/my_run all
   ```
   Log: `slurm/logs/linguistic_run_<jobid>.out`.
6. Copy `out/my_run/` back.

Notes:
- Use the `pi_evelina9` or `ou_bcs_normal` partitions.
- GPT-2 XL needs about 7 GB of RAM. A GPU is optional for a few hundred stimuli.
- GPU and CPU results differ slightly (about the 5th decimal).
