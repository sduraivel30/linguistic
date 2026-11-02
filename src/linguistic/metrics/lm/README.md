# Language-model metrics

Word level, context-dependent. Each stimulus is scored on its own; words are scored as
written (case kept), so lowercase the `text` if needed. One metric per model, best of each
family on the brain benchmarks of Schrimpf et al. (2021):

| Metric | Model | Family | The word is scored given |
| --- | --- | --- | --- |
| `lm_gpt2-xl` | [gpt2-xl](https://huggingface.co/openai-community/gpt2-xl) | decoder-only | preceding words |
| `lm_xlnet-large-cased` | [xlnet-large-cased](https://huggingface.co/xlnet/xlnet-large-cased) | permutation LM | preceding words |
| `lm_albert-xxlarge-v1` | [albert-xxlarge-v1](https://huggingface.co/albert/albert-xxlarge-v1) | encoder-only | words on both sides |
| `lm_bert-large-uncased` | [bert-large-uncased](https://huggingface.co/google-bert/bert-large-uncased) | encoder-only | words on both sides |
| `lm_t5-large` | [t5-large](https://huggingface.co/google-t5/t5-large) | encoder-decoder | words on both sides |

Columns per model (`<tag>` = `gpt2xl`, `xlnetlarge`, `albertxxl`, `bertlarge`, `t5large`):

| Column | Definition |
| --- | --- |
| `lm_surprisal_<tag>` | mean surprisal over the word's sub-tokens (nats) |
| `lm_entropy_<tag>` | entropy of the model's prediction for the word's first sub-token (nats) |
| `lm_n_subtokens_<tag>` | number of sub-tokens |

Encoder-only models: the word is masked and predicted left to right within the word
(whole-word pseudo-log-likelihood, Kauf & Ivanova 2023). T5: the word is replaced by a
blank (`<extra_id_0>`) and the model fills it in.
XLNet reads every stimulus after a fixed padding passage (the standard one from the Hugging
Face XLNet examples), because it predicts poorly from a short context.

## Activations and embeddings

```bash
linguistic represent -i stimuli.csv -o reps/ -m gpt2-xl -m albert-xxlarge-v1
```

Per model, in `reps/<model>/`: `activations.npy` (words × layers × units, the embedding
layer then every block), `embeddings.npy` (the word's input embedding, no context),
`stimuli_last.npy` and `stimuli_mean.npy` (per stimulus: last word, mean over words), and
`words.csv` / `stimuli.csv` / `layers.csv` giving the rows. By default each word is read
with the preceding words only (`--context full` for the whole stimulus); a word's vector is
its last sub-token (`--pool mean` for the mean). `--dtype float16` halves the size. T5:
encoder layers.

## Setup

Models are not in the repo. Each is pinned to a revision and goes in
`$LINGUISTIC_RESOURCES/<model>/` (default `~/.cache/linguistic`):

```bash
pip install -e ".[lm]"
linguistic fetch gpt2-xl        # 6.4 GB; also xlnet-large-cased, albert-xxlarge-v1, bert-large-uncased, t5-large
```

GPU recommended. On a cluster see [`slurm/README.md`](../../../../slurm/README.md). To add
a model, add a line to `MODELS` in `models.py`.

## Run

```bash
linguistic run -i stimuli.csv -o out/ -m lm
```
