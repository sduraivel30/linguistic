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
   for m in gpt2-xl xlnet-large-cased albert-xxlarge-v1 bert-large-uncased t5-large; do
     srun -p <partition> -t 00:30:00 --mem 8G .venv/bin/linguistic fetch $m
   done
   ```
4. Copy your stimulus file to the cluster, e.g. into `data/`.
5. Run (drop `--gres` to run on CPU):
   ```bash
   sbatch -p <partition> --gres=gpu:1 slurm/run_metrics.sh data/my_stimuli.csv out/my_run all
   ```
   Activations and embeddings:
   ```bash
   sbatch -p <partition> --gres=gpu:1 slurm/represent.sh data/my_stimuli.csv reps/my_run gpt2-xl albert-xxlarge-v1
   ```
   Logs: `slurm/logs/`.
6. Copy `out/my_run/` (and `reps/my_run/`) back.

Notes:
- Downloads: GPT-2 XL 6.4 GB, T5-large 3 GB, the others 1–1.5 GB. A GPU is optional for a
  few hundred stimuli.
- GPU and CPU results differ slightly (about the 5th decimal).
