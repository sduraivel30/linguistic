# linguistic

Extract linguistic metrics for a list of sentences or nonwords, at the phone, word and
stimulus level. Setup plan: https://claude.ai/artifact/Up1mTvhWmRzu9ePv13TBYa

## Install

```bash
python -m venv .venv && .venv/bin/pip install -e ".[dev]"
```

## Input

One CSV/TSV, one row per stimulus:

| Column | Required | Meaning |
| --- | --- | --- |
| `stimulus_id` | yes | unique id |
| `condition` | yes | `sentence` or `nonword` |
| `text` | yes | the stimulus, space-separated units |
| `words_labels` | no | Audacity label file of word intervals |
| `phones_labels` | no | Audacity label file of phone intervals (ARPAbet) |
| `labels` | no | one Audacity file with both tiers (uppercase labels = phones, rest = words), instead of the pair above |
| `audio_path` | no | only for acoustic metrics |

Relative paths resolve against the stimulus file's folder. Hand labels come as a
words + phones pair; each phone goes to the word it overlaps most in time, and word
bounds are then taken from the phones. A hyphenated text word split into its parts
in the labels (`ice` + `cream` for `ice-cream`) is merged back into one word.

## Phones

1. Hand labels, when given.
2. Otherwise CMUdict, if the spelling is in it (first pronunciation, stress stripped).
   Every word and nonword is tagged `in_lexicon`.
3. Otherwise none: phone-based metrics are NaN and the word is listed in
   `missing_phones.csv` until hand phones are supplied.

No grapheme-to-phoneme prediction is used.

## Run

```bash
linguistic list                                   # registered metrics
linguistic prepare -i stimuli.csv -o out/         # word/phone tables + missing phones only
linguistic run -i stimuli.csv -o out/ -m all      # or family / metric names, repeatable
```

Outputs: `words.csv`, `phones.csv`, `stimuli.csv`, `missing_phones.csv`, `manifest.json`.

- A phone-level metric appears in `words.csv` as the mean over the word's phones.
- `stimuli.csv` averages every numeric word-level metric over the stimulus's words
  (NaN words skipped, `n_valid_<metric>` counts), so phone-level metrics are a mean of
  per-word means.
- Values are NaN where a metric's declared conditions or requirements (phones, timing)
  are not met.
- Read the CSVs with `keep_default_na=False, na_values=['']`: some nonwords (`nan`) are
  otherwise read as missing.

## Metric families

Each family is a folder under `src/linguistic/metrics/` with its own `README.md`: metric
definitions, source, how to download and set up any model or lexicon it needs, and its
regression check.

| Family | README |
| --- | --- |
| phonological | [src/linguistic/metrics/phonological/README.md](src/linguistic/metrics/phonological/README.md) |

## Adding a metric

Subclass `linguistic.registry.Metric` in a module under `src/linguistic/metrics/<family>/`,
decorate it with `@register`, import it in the family's `__init__.py`, and import the
family in `metrics/__init__.py`.
See the docstring in `registry.py` for the contract.

## Tests

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest
```
