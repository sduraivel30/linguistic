# Running on a server (SLURM)

Step-by-step notes for running the pipeline on a SLURM cluster, written for MIT ORCD
(`orcd-login.mit.edu`) and usable elsewhere by changing partitions and paths. Language
models are the main reason to use a server: they need several GB of weights, RAM and
ideally a GPU. Everything else runs fine on a laptop.

Two scripts:

| Script | What it does |
| --- | --- |
| `setup_env.sh` | one-time: builds `.venv` in the repo with the language-model extras, downloads CMUdict and SUBTLEX-US |
| `run_metrics.sh` | runs `linguistic run` on a stimulus file, as a batch job |

## Rules that save time

- **All compute goes through `sbatch` (or `srun`), never on the login node.** That
  includes `pip install` of large packages and any model run.
- Submit from the **repo root**. The scripts `cd "$SLURM_SUBMIT_DIR"`. Under sbatch,
  `dirname "$0"` points at SLURM's spooled copy of the script, not the repo.
- **Create the log folder before submitting**: `mkdir -p slurm/logs`. SLURM does not
  create it, and the job fails without writing a log.
- `/tmp` on a compute node is local to that node. Put anything a later step needs on
  shared storage (the repo, `$LINGUISTIC_RESOURCES`, an output folder under `/orcd/...`).
- `export LINGUISTIC_RESOURCES=...` before `sbatch`: the scripts are submitted with the
  current environment and stop if it is unset.
- ORCD partitions: `pi_evelina9` (lab node, 8× A100, 7-day limit, starts immediately) and
  `ou_bcs_normal` (1-day limit). Avoid `mit_normal`, which has no priority for this group.
- ORCD compute nodes have internet access, so downloads (pip, Hugging Face) work inside jobs.
- ORCD prints a QUOTA banner on every ssh login, including non-interactive ones. It ends
  up inside anything you pull back with `ssh ... cat file > local`. Strip it, or pull with
  markers: `ssh host 'echo @@START; base64 < file; echo @@END'`, then keep only the lines
  between the markers and `base64 -D`.

## Step by step

### 1. Get the code onto the server (once)

```bash
ssh orcd-login.mit.edu
cd /orcd/data/evelina9/001/USERS/sdurai30/repos
git clone https://github.com/sduraivel30/linguistic.git
cd linguistic
```

Later updates: `git pull` in that folder. Never `git stash` on the cluster: if the tree
is dirty, check out single files instead.

### 2. Choose a shared resource folder (once)

Lexicons and models live outside the repo, in one folder per resource:

```bash
export LINGUISTIC_RESOURCES=/orcd/data/evelina9/001/USERS/sdurai30/data/linguistic_resources
mkdir -p $LINGUISTIC_RESOURCES
```

Add the `export` line to `~/.bashrc` so every login and job sees it.

### 3. Build the environment (once, as a job)

```bash
mkdir -p slurm/logs
sbatch slurm/setup_env.sh
```

This loads `miniforge/26.7.2-0` (override with `MINIFORGE_MODULE=...`), creates
`.venv`, installs the package with `[lm,dev]` (torch, transformers, huggingface_hub,
pytest), downloads CMUdict and SUBTLEX-US, and prints the torch/CUDA versions and the
metric list. Check `slurm/logs/linguistic_setup_<jobid>.out`.

### 4. Get each language model (once per model)

A model must end up as a folder `$LINGUISTIC_RESOURCES/<model name>/` holding
`config.json`, the tokenizer files and `model.safetensors`, at the revision pinned in
`src/linguistic/metrics/lm/causal.py`. Two ways:

**a. Download it, as a job** (compute nodes have internet):

```bash
srun -p pi_evelina9 -t 00:30:00 --mem 8G .venv/bin/linguistic fetch gpt2-xl
```

**b. Reuse a copy already in a Hugging Face cache**, if it is the same revision. Link the
snapshot folder instead of downloading again:

```bash
ls ~/.cache/huggingface/hub/models--gpt2-xl/snapshots/   # or your HF_HOME/hub/...
ln -s <hf cache>/hub/models--gpt2-xl/snapshots/15ea56dee5df4983c59b2538573817e1667135e2 \
      $LINGUISTIC_RESOURCES/gpt2-xl
ls -L $LINGUISTIC_RESOURCES/gpt2-xl     # config.json generation_config.json merges.txt
                                        # model.safetensors tokenizer.json tokenizer_config.json vocab.json
```

The snapshot folder name is the revision: it must match the pinned one.

Sizes and resources per model:

| Model (`-m` name) | Download | RAM (float32) | GPU |
| --- | --- | --- | --- |
| `lm_gpt2-xl` | 6.4 GB | about 7 GB | optional: a few hundred short stimuli take minutes on CPU, seconds on one GPU |

### 5. Put the stimulus file on the server

From your own machine:

```bash
cat my_stimuli.csv | ssh orcd-login.mit.edu \
  'cat > /orcd/data/evelina9/001/USERS/sdurai30/repos/linguistic/data/my_stimuli.csv'
```

(`data/` is git-ignored except `data/examples/`.) Hand-label paths in the file must be
valid on the server.

### 6. Run

```bash
cd /orcd/data/evelina9/001/USERS/sdurai30/repos/linguistic
sbatch --gres=gpu:1 slurm/run_metrics.sh data/my_stimuli.csv out/my_run lm_gpt2-xl
# everything: sbatch --gres=gpu:1 slurm/run_metrics.sh data/my_stimuli.csv out/my_run all
squeue -u $USER                           # pending / running
sacct -j <jobid> -o JobID,State,Elapsed,MaxRSS
cat slurm/logs/linguistic_run_<jobid>.out
```

Without `--gres=gpu:1` the job runs on CPU. The log starts with the GPU name, or
`no GPU: running on CPU`. For a chain of jobs (e.g. run, then a summary), use
`sbatch --dependency=afterok:<jobid> ...` rather than submitting the second by hand.

### 7. Bring the results back

The outputs are small CSVs plus `manifest.json`. Pull them with the marker method so the
QUOTA banner is removed:

```bash
mkdir -p my_run
ssh orcd-login.mit.edu 'cd /orcd/data/evelina9/001/USERS/sdurai30/repos/linguistic/out/my_run \
  && echo @@START && tar czf - . | base64 && echo @@END' \
  | sed -n '/^@@START$/,/^@@END$/p' | sed '1d;$d' | base64 -D | tar xzf - -C ./my_run
```

Check `manifest.json`: it records the model revision, package versions and the code
commit used.

## Language-model specifics

- Models are loaded with `local_files_only=True`: a run never downloads. If a model is
  missing, `-m all` skips it and says so, and naming it (`-m lm_gpt2-xl`) fails with
  the path it expected.
- Device: CUDA if visible, else Apple MPS, else CPU. Force one with
  `export LINGUISTIC_DEVICE=cpu` (or `cuda`) before `sbatch`.
- Precision is float32. Values from a GPU and from a CPU differ in about the 5th decimal
  place: compare runs with a tolerance, not exact equality.
- Each stimulus is scored as its own context, so memory depends on the longest stimulus
  and the model, not on the number of stimuli.
- Adding a model: add one line to `MODELS` in `src/linguistic/metrics/lm/causal.py`
  (name, Hugging Face repo, pinned revision, column tag), then step 4 for that model. Models
  of 7B parameters and up need a GPU (about 28 GB in float32) and, for gated models such as
  Llama, a Hugging Face token: `hf auth login` once on the server, after accepting the
  licence on the model page.
