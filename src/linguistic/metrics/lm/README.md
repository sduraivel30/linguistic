# Language-model metrics

Word level, context-dependent. Each stimulus is scored on its own, starting from a BOS
token; words are scored as written (case kept), so lowercase the `text` if needed.

| Metric | Columns | Definition |
| --- | --- | --- |
| `lm_gpt2-xl` | `lm_surprisal_gpt2xl` | mean surprisal over the word's sub-tokens (nats) |
| | `lm_entropy_gpt2xl` | entropy of the prediction just before the word (nats) |
| | `lm_n_subtokens_gpt2xl` | number of sub-tokens |

## Setup

GPT-2 XL ([openai-community/gpt2-xl](https://huggingface.co/openai-community/gpt2-xl),
revision `15ea56d`, 6.4 GB) is not in the repo:

```bash
pip install -e ".[lm]"
linguistic fetch gpt2-xl          # into $LINGUISTIC_RESOURCES/gpt2-xl (default ~/.cache/linguistic)
```

GPU optional. On a cluster see [`slurm/README.md`](../../../../slurm/README.md).

To add a model, add a line to `MODELS` in `causal.py` (name, Hugging Face repo, revision,
column tag).

## Run

```bash
linguistic run -i stimuli.csv -o out/ -m lm
```
