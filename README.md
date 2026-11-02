# linguistic

Linguistic metrics for sentence and nonword lists, at the phone, word and stimulus level.

## Install

```bash
python -m venv .venv && .venv/bin/pip install -e ".[dev,lm]"
linguistic fetch all          # lexicons and models with a public download
```

## Input

CSV with `stimulus_id`, `condition` (`sentence` / `nonword`) and `text`. Optional hand
labels (Audacity format): `words_labels` + `phones_labels`, or one `labels` file with both
tiers. Optional `audio_path`.

Phones come from the hand labels, else from CMUdict. Words with neither are listed in
`missing_phones.csv` and their phone-based metrics are NaN.

## Run

```bash
linguistic list                                   # available metrics
linguistic run -i stimuli.csv -o out/ -m all      # or a family, a metric, or level:context
```

Outputs: `phones.csv`, `words.csv`, `stimuli.csv`, `columns.csv` (what each column is),
`missing_phones.csv`, `manifest.json`. Word values of phone metrics are means over the
word's phones; stimulus values are means over its words.

Examples: [`data/examples/`](data/examples/).

## Metrics

Each column is context-independent (depends only on the unit) or context-dependent
(depends on a phone or word sequence) at each level.

| Level | Context-independent | Context-dependent |
| --- | --- | --- |
| phone | phonetic features, single-phone BLICK | phoneme transitional probability, BLICK windows |
| word | feature shares, word duration, word frequency | phoneme TP, BLICK, cohort density, uniqueness point, LM surprisal and entropy |
| stimulus | means of the word-level columns | means of the word-level columns |

Definitions and setup per family:
[phonological](src/linguistic/metrics/phonological/README.md) ·
[lexical](src/linguistic/metrics/lexical/README.md) ·
[lm](src/linguistic/metrics/lm/README.md).

Running on a SLURM cluster: [`slurm/README.md`](slurm/README.md).

## Tests

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest
```
