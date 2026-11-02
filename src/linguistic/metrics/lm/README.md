# Language-model metrics

Word-level metrics from causal (left-to-right) language models. One metric per model;
GPT-2 XL is the only model registered so far. Scored for sentences and nonwords (nonwords
are scored as strings, through the model's sub-word tokens). In `stimuli.csv` each is the
mean over the stimulus's words.

**Context:** context-dependent at the word and stimulus level: every value is conditioned
on the preceding words of the stimulus.

## Metrics

| Metric (`-m` name) | Columns | Definition |
| --- | --- | --- |
| `lm_gpt2-xl` | `lm_surprisal_gpt2xl` | mean over the word's sub-tokens of −ln p(sub-token \| all preceding tokens of the stimulus), nats |
| | `lm_entropy_gpt2xl` | entropy, in nats, of the model's next-token distribution at the position that predicts the word's first sub-token: its uncertainty just before the word |
| | `lm_n_subtokens_gpt2xl` | number of GPT-2 sub-tokens in the word |

Summed (whole-word) surprisal is not reported: it grows with the number of sub-tokens.

How a stimulus is scored:

- Context resets at every stimulus. A BOS token (GPT-2: its end-of-text token) comes
  before word 1, so word 1 has a defined surprisal.
- The model sees the stimulus's words joined by single spaces, **case kept**,
  punctuation at word edges removed. ALL-CAPS input (like the EvLab examples) is scored as
  ALL-CAPS text, which GPT-2 finds much less predictable. Lowercase or sentence-case your
  `text` column if that is not what you want.
- Words after the first are encoded with their leading space. The word-by-word encoding
  must equal the whole-string encoding (asserted). The mean token loss must match the
  model's own loss within 2e-3 (asserted).
- Float32. Device: CUDA if available, else Apple MPS, else CPU; override with
  `LINGUISTIC_DEVICE=cpu`.

## Get the model

The model is not stored in the repo. GPT-2 XL
([openai-community/gpt2-xl](https://huggingface.co/openai-community/gpt2-xl), revision
`15ea56d`, 6.4 GB) goes in `$LINGUISTIC_RESOURCES/gpt2-xl/` (default `~/.cache/linguistic`).

```bash
pip install -e ".[lm]"
linguistic fetch gpt2-xl
```

To place it by hand, download `config.json`, `generation_config.json`, `tokenizer.json`,
`tokenizer_config.json`, `vocab.json`, `merges.txt` and `model.safetensors` from the model
page at that revision into the same folder. Runs never download anything; without the
model, `-m all` skips this metric. On ORCD, see [`slurm/README.md`](../../../../slurm/README.md).

## Adding another model

Add one line to `MODELS` in `causal.py`: a name, the Hugging Face repo, a pinned revision
(the commit hash on the model page), and a short column tag. The metric `lm_<name>`, its
columns and its `linguistic fetch <name>` download then exist automatically. Values are
comparable across models only roughly: each model has its own tokenizer, so sub-token
counts and per-sub-token surprisal differ.

## Run

```bash
linguistic run -i stimuli.csv -o out/ -m lm            # every registered model that is installed
linguistic run -i stimuli.csv -o out/ -m lm_gpt2-xl
```
