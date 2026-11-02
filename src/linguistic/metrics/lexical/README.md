# Lexical metrics

Word level; stimulus value = mean over its words.

| Metric | Columns | Context | Conditions | Definition |
| --- | --- | --- | --- | --- |
| `subtlex_frequency` | `subtlex_lg10wf`, `subtlex_zipf` | independent | sentence | SUBTLEX-US log10(count + 1); Zipf (van Heuven et al. 2014) |
| `elp_frequency` | `elp_log_freq_hal` | independent | sentence | ELP log HAL frequency |
| `word_duration` | `word_duration_ms` | independent | both | word offset − onset (needs hand-label timing) |
| `cohort` | `cohort_density_log10` | dependent | both | log10(1 + Σ number of CMUdict words sharing the first k phonemes) |
| | `uniqueness_point_idx`, `uniqueness_point_ms` | dependent | both | first phoneme where no other word shares the prefix (else the last); ms needs timing |
| | `up_reached` | dependent | both | 1 if the word becomes unique before its end |

Frequency lookups use the lowercase spelling (hyphenated words also tried without the
hyphen). Missing words are NaN.

## Setup

```bash
linguistic fetch subtlex_us
```

ELP: export `Word` and `Log_Freq_HAL` from https://elexicon.wustl.edu/ and save as
`$LINGUISTIC_RESOURCES/elp/elp_words.csv` (default `~/.cache/linguistic`). Without it,
`elp_frequency` is skipped.

## Run

```bash
linguistic run -i stimuli.csv -o out/ -m lexical
```
