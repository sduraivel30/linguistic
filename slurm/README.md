# Running on a SLURM cluster

Mainly for the language models. Run from the repo root.

1. Clone the repo and pick a shared folder for models and lexicons (put the export in
   `~/.bashrc`):
   ```bash
   git clone https://github.com/sduraivel30/linguistic.git && cd linguistic
   export LINGUISTIC_RESOURCES=/path/to/shared/linguistic_resources
   mkdir -p $LINGUISTIC_RESOURCES slurm/logs
   ```
2. Build the environment (once). Set `PYTHON_MODULE` to your cluster's Python module, or
   leave it unset to use the `python3` on `PATH`:
   ```bash
   sbatch -p <partition> slurm/setup_env.sh
   ```
3. Download each model you need (once):
   ```bash
   srun -p <partition> -t 00:30:00 --mem 8G .venv/bin/linguistic fetch gpt2-xl
   ```
4. Copy your stimulus file to the cluster, e.g. into `data/`.
5. Run (drop `--gres` to run on CPU):
   ```bash
   sbatch -p <partition> --gres=gpu:1 slurm/run_metrics.sh data/my_stimuli.csv out/my_run all
   ```
   Log: `slurm/logs/linguistic_run_<jobid>.out`.
6. Copy `out/my_run/` back.

Notes:
- GPT-2 XL needs about 7 GB of RAM. A GPU is optional for a few hundred stimuli.
- GPU and CPU results differ slightly (about the 5th decimal).
