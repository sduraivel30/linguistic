# linguistic

Linguistic metrics for sentence and nonword lists. Every metric is reported at three
levels: per **phone**, per **word** (word or nonword) and per **stimulus** (sentence or
nonword list), in `phones.csv`, `words.csv` and `stimuli.csv`.

## Install

```bash
python -m venv .venv && .venv/bin/pip install -e ".[dev,lm]"
linguistic fetch all          # lexicons and models with a public download
```

## Input

CSV with `stimulus_id`, `condition` (`sentence` / `nonword`) and `text`. Optional hand
labels (Audacity format): `words_labels` + `phones_labels`, or one `labels` file with both
tiers.

Phones come from the hand labels, else from CMUdict. Words with neither are listed in
`missing_phones.csv` and their phone-based metrics are NaN.

## Run

```bash
linguistic list                                   # available metrics
linguistic run -i stimuli.csv -o out/ -m all      # or a family, a metric, or level:context
linguistic represent -i stimuli.csv -o reps/ -m gpt2-xl     # layer activations, embeddings
```

`columns.csv` in each output lists every column with its level and context.
Examples: [`data/examples/`](data/examples/).

## What is computed at each level

A metric is defined at one level and carried up one level: a phone value to the word (mean
over the word's phones, never across words), a word value to the stimulus (mean over its
words). Context-independent (I) means the value depends only on that unit;
context-dependent (D) means it depends on a sequence of phones or words.

| Metric | Phone | Word | Stimulus |
| --- | --- | --- | --- |
| phonetic features (`feat_*`) | I: 1/0 per feature | I: share of phones | – |
| single-phone BLICK (`blick_w1_*`) | I: score | I: mean over phones | – |
| phoneme transitional probability (`*_surp_cross/_within/_back`) | D: surprisal | D: mean over phones | – |
| BLICK windows (`blick_w2-4_*`, `delta_*`) | D: score | D: mean over phones | – |
| cohort surprisal, entropy (`cohort_*`) | D: value | D: mean over phones | – |
| BLICK whole word (`blick_word`) | – | D: score | D: mean over words |
| cohort density, uniqueness point | – | D: value | D: mean over words |
| word frequency (SUBTLEX, ELP) | – | I: value | I: mean over words |
| word duration | – | I: value | I: mean over words |
| LM surprisal, entropy (`lm_*`) | – | D: value | D: mean over words |
| 5-gram surprisal (`ngram_*`, sentences) | – | D: value | D: mean over words |
| LM activations (`represent`) | – | D: per layer | D: last word, mean over words |
| LM input embeddings (`represent`) | – | I: vector | – |
| GloVe embeddings (`represent -m glove-840b`) | – | I: vector | I: last word, mean over words |

Definitions and setup per family:
[phonological](src/linguistic/metrics/phonological/README.md) ·
[lexical](src/linguistic/metrics/lexical/README.md) ·
[lm](src/linguistic/metrics/lm/README.md).

Running on a SLURM cluster: [`slurm/README.md`](slurm/README.md).

## Known limitations

- English only; phones are ARPAbet. No grapheme-to-phoneme prediction: words outside
  CMUdict need hand labels.
- ELP frequency needs an ELP export placed by hand.
- BLICK gives every vowel primary stress, real words included.
- Text is scored as written: case matters for the language models.
- XLNet reads each stimulus after a fixed padding passage.
- The 5-gram model needs Apptainer, and scores sentences only.
- The language models need a GPU for large stimulus sets.

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md). Licence: MIT. Cite with [`CITATION.cff`](CITATION.cff).

## Tests

```bash
python -m pytest
```

Language-model tests use small test models from the local Hugging Face cache and are
skipped when those are not downloaded.
