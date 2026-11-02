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

### BLICK

BLICK (Hayes) scores a phone string with weighted phonotactic constraints and returns a
penalty: 0 = fully English-like, higher = less English-like (`blick` 0.116, `bnick` 5.95).
Every vowel gets primary stress (`IH1`), because stress is undefined for nonwords.

| Metric (`-m` name) | Level | Columns | Definition |
| --- | --- | --- | --- |
| `blick_word` | word | `blick_word` | BLICK score of the whole word's phones |
| `blick_cross` | phone | `blick_w1_cross` … `blick_w4_cross` | BLICK score of the last N phones ending at this phone (N = 1–4; fewer at the start), window running across word boundaries |
| | | `delta_blick_w2_cross` … `delta_blick_w4_cross` | change in the N-phone score from the previous phone (NaN at a sequence's first phone) |
| `blick_within` | phone | `blick_w1_within` … `blick_w4_within`, `delta_blick_w2_within` … `delta_blick_w4_within` | same, window reset at each word boundary (delta NaN at a word's first phone) |

## Source

Vendored from `word_form_boundary_analysis` @ `33e104f` (`src/tp_analysis/corpus.py`,
`src/tp_analysis/blick_analysis.py`, `scripts/run_tp_blick_analysis.py`) and
`Speech_Lang_iEEG` @ `1ec9f82` (`surprisal/phoneme/build_backward_values.py`). Changes:
counts are kept in ARPAbet rather than IPA (the source's ARPAbet→IPA map is one-to-one, so
values are identical); a word without phones breaks cross-word contexts and windows;
`blick_word` is new.

## Resources and setup

Two resources, both installed with this package.

**CMUdict** (phone counts, phone lookup), via nltk:

1. `pip install -e .` (pulls in `nltk`).
2. Download CMUdict once:
   ```bash
   python -c "import nltk; nltk.download('cmudict')"
   ```
   It goes to `~/nltk_data/corpora/cmudict`; the first run downloads it automatically if
   missing. Source: http://www.speech.cs.cmu.edu/cgi-bin/cmudict

**BLICK**, via python-BLICK 0.2.12 (McAuliffe & Hayes; pure Python, no model files):

1. `pip install -e .` installs it (pinned in `pyproject.toml`). By hand:
   `pip install python-BLICK==0.2.12`
2. Check: `python -c "from blick import BlickLoader; print(BlickLoader().assessWord('B L IH1 K'))"`
   prints `0.116`.
3. Links: package https://pypi.org/project/python-BLICK/0.2.12/ · code
   https://github.com/mmcauliffe/python-BLICK · original BLICK (Hayes)
   http://www.linguistics.ucla.edu/people/hayes/BLICK/

Then `linguistic list` should show six `phonological` metrics.

## Run

```bash
linguistic run -i stimuli.csv -o out/ -m phonological          # all six
linguistic run -i stimuli.csv -o out/ -m phoneme_tp_cross       # just one
linguistic run -i stimuli.csv -o out/ -m blick_word -m blick_cross
```

## Regression check

`scripts/regression/phonological_vs_langloc.py` runs these metrics on the 114 langloc items
(canonical MT transcriptions, 6 excluded items left out) and compares them with
`Speech_Lang_iEEG/surprisal/phoneme/tp_blick_per_phone.csv` (repaired) and
`plots/backward_values_<cond>.npz`, matching phones by item, start time and label.

Result (2026-11-02): 5045 / 5045 phones matched. Phone labels, word assignment
(`word_idx`, `phone_in_word`) and the NaN pattern are identical. All 8 n-gram columns and
all 14 BLICK columns agree within 1e-14. The reference has the within-word BLICK columns
blank for the phones whose word assignment was repaired on 2026-10-21; this port fills them
(335 window values and 232 deltas in these 114 items). `blick_word` has no reference
(new); a unit test checks it equals the 4-phone within-word window for words of up to 4 phones.

```bash
python scripts/regression/phonological_vs_langloc.py <out dir>     # exits 1 on any difference
```
