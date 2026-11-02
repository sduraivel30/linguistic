# Phonological metrics

Phone-level metrics computed from each word's phones (hand labels, else CMUdict; words
without phones are NaN). Scored for sentences and nonwords. In `words.csv` each appears as
the mean over the word's phones; in `stimuli.csv` as the mean of those per-word means.

## Metrics

All logs are natural logs (surprisal in nats). Counts come from every CMUdict
pronunciation (133,737 entries, stress stripped), counting n-grams within words. N-grams
never seen in CMUdict give NaN.

| Metric (`-m` name) | Columns | Definition, for a phone sequence ... A B C ... |
| --- | --- | --- |
| `phoneme_tp_cross` | `bigram_surp_cross` | −ln P(B \| A) = −ln[count(A,B) / count(A)], context carried across word boundaries |
| | `trigram_surp_cross` | −ln P(C \| A,B) = −ln[count(A,B,C) / count(A,B)] |
| | `log_bigram_freq_cross` | ln count(A,B) |
| `phoneme_tp_within` | `bigram_surp_within`, `trigram_surp_within`, `log_bigram_freq_within` | same, but context resets at each word boundary: the first phone of a word is NaN for bigram, the first two for trigram |
| `phoneme_tp_backward` | `bigram_surp_back` | −ln P(B \| C) = −ln[count(B,C) / count(C)], assigned to B, across word boundaries |
| | `trigram_surp_back` | −ln P(X \| Y,Z) = −ln[count(X,Y,Z) / count(Y,Z)], assigned to X |

The first phone of a stimulus has no forward value; the last one (two for trigram) has no
backward value. A word without phones breaks the cross-word context on both sides, so no
transition is ever scored across a missing word.

## Source

Vendored from `word_form_boundary_analysis` @ `33e104f` (`src/tp_analysis/corpus.py`,
`scripts/run_tp_blick_analysis.py`) and `Speech_Lang_iEEG` @ `1ec9f82`
(`surprisal/phoneme/build_backward_values.py`). The only changes: counts are kept in ARPAbet
rather than IPA (the source's ARPAbet→IPA map is one-to-one, so values are identical), and
the missing-word context break above.

## Resources and setup

Only CMUdict is needed, via nltk (installed with this package).

1. Install the package: `pip install -e .` (pulls in `nltk`).
2. Download CMUdict once:
   ```bash
   python -c "import nltk; nltk.download('cmudict')"
   ```
   It goes to `~/nltk_data/corpora/cmudict`. If it's missing, the first run downloads it
   automatically. Source: http://www.speech.cs.cmu.edu/cgi-bin/cmudict
3. Check: `linguistic list` should show the three `phonological` metrics.

## Run

```bash
linguistic run -i stimuli.csv -o out/ -m phonological          # all three
linguistic run -i stimuli.csv -o out/ -m phoneme_tp_cross       # just one
```

## Regression check

`scripts/regression/phonological_vs_langloc.py` runs these metrics on the 114 langloc items
(canonical MT transcriptions, 6 excluded items left out) and compares them with
`Speech_Lang_iEEG/surprisal/phoneme/tp_blick_per_phone.csv` (repaired) and
`plots/backward_values_<cond>.npz`, matching phones by item, start time and label.

Result (2026-11-02): 5045 / 5045 phones matched. Phone labels, word assignment
(`word_idx`, `phone_in_word`) and the NaN pattern are identical, and all 11 value columns
agree within 2e-15.
