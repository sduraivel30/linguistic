# Phonological metrics

Metrics computed from each word's phones (hand labels, else CMUdict; words without phones
are NaN). Scored for sentences and nonwords. Phone-level metrics appear in `words.csv` as
the mean over the word's phones and in `stimuli.csv` as the mean of those per-word means.

**Context** (per level, also listed in each run's `columns.csv`): *independent* = depends
only on the unit itself; *dependent* = depends on a sequence of phones (preceding phones,
the word's other phones, or neighbouring words). The stimulus level takes the word-level
label.

## Context-independent

### Phonetic features

14 binary features in the standard intracranial-speech convention (Mesgarani et al. 2014,
*Science* 343:1006–1010). Memberships follow the TIMIT feature table in
https://github.com/ChangLabUcsf/HamiltonOganianChang2020 (`util1/Timit_features.mat`),
restricted to the 39 ARPAbet phones used here; the table is written out in `features.py`.

| Metric (`-m` name) | Columns | Phone | Word | Definition |
| --- | --- | --- | --- | --- |
| `phonetic_features` | `feat_<feature>` | independent | independent | phone: 1 if the phone has the feature, else 0. Word: share of the word's phones that have it |

| Group | Feature | ARPAbet phones |
| --- | --- | --- |
| place | `dorsal` | CH G JH K NG SH |
| | `coronal` | CH D JH N S T Z |
| | `labial` | B F M P V |
| vowel position | `high` | EH EY IH IY UH UW |
| | `front` | AE EH EY IH IY |
| | `low` | AA AE AH AO AW AY |
| | `back` | AA AH AO OW UH UW |
| manner | `plosive` | B D G K P T |
| | `fricative` | CH DH F HH JH S SH TH V Z ZH |
| | `syllabic` | AA AE AH AO AW AY EH ER EY IH IY OW OY UH UW |
| | `nasal` | M N NG |
| voicing / major class | `voiced` | AA AE AH AO AW AY B D DH EH EY G IH IY L M N NG OW OY R UH UW V W Y Z |
| | `obstruent` | B CH D DH F G HH JH K P S SH T TH V Z ZH |
| | `sonorant` | AA AE AH AO AW AY EH EY IH IY L M N NG OW OY R UH UW W Y |

A phone belongs to every feature it is listed under (`T`: coronal, plosive, obstruent).
By the same convention, `TH`, `DH`, `L`, `R`, `W`, `Y` and `HH` have no place feature.

> **Note: kept as published.** Four memberships differ from textbook phonetics and are
> kept so values match the published feature set: `ER`, `JH` and `ZH` are not `voiced`,
> `ER` is not `sonorant`, and `SH` is `dorsal` while `ZH` is not.

### Single-phone BLICK

| Metric | Columns | Phone | Word | Definition |
| --- | --- | --- | --- | --- |
| `blick_cross`, `blick_within` | `blick_w1_cross`, `blick_w1_within` | independent | independent | BLICK score of the phone on its own (the two columns are identical) |

> **Disclaimer: single-phone BLICK.** BLICK scores any input as if it were a whole word.
> For one phone this mostly measures whether that phone could be a word by itself: a lone
> consonant is penalised for having no vowel (`T` scores 15.6). Treat it as a phone-identity
> value, not a measure of phonotactic probability.

## Context-dependent

All logs are natural logs (surprisal in nats). N-gram counts come from every CMUdict
pronunciation (133,737 entries, stress stripped), counting n-grams within words. N-grams
never seen in CMUdict give NaN.

### Phoneme transitional probability

| Metric (`-m` name) | Columns | Phone | Word | Definition, for a phone sequence … A B C … |
| --- | --- | --- | --- | --- |
| `phoneme_tp_cross` | `bigram_surp_cross` | dependent | dependent | −ln P(B \| A) = −ln[count(A,B) / count(A)], context carried across word boundaries |
| | `trigram_surp_cross` | | | −ln P(C \| A,B) = −ln[count(A,B,C) / count(A,B)] |
| | `log_bigram_freq_cross` | | | ln count(A,B) |
| `phoneme_tp_within` | `bigram_surp_within`, `trigram_surp_within`, `log_bigram_freq_within` | dependent | independent | same, but context resets at each word boundary: the first phone of a word is NaN for bigram, the first two for trigram |
| `phoneme_tp_backward` | `bigram_surp_back` | dependent | dependent | −ln P(B \| C) = −ln[count(B,C) / count(C)], assigned to B, across word boundaries |
| | `trigram_surp_back` | | | −ln P(X \| Y,Z) = −ln[count(X,Y,Z) / count(Y,Z)], assigned to X |

The first phone of a stimulus has no forward value; the last one (two for trigram) has no
backward value. A word without phones breaks the cross-word context on both sides, so no
transition is ever scored across a missing word.

### BLICK (phone sequences)

BLICK (Hayes) scores a phone string with weighted phonotactic constraints and returns a
penalty: 0 = fully English-like, higher = less English-like (`blick` 0.116, `bnick` 5.95).
It works on the phoneme sequence, so every multi-phone score is context-dependent. Every
vowel gets primary stress (`IH1`), because stress is undefined for nonwords.

| Metric (`-m` name) | Columns | Phone | Word | Definition |
| --- | --- | --- | --- | --- |
| `blick_word` | `blick_word` | – | dependent | BLICK score of the whole word's phone sequence |
| `blick_cross` | `blick_w2_cross` … `blick_w4_cross` | dependent | dependent | BLICK score of the last N phones ending at this phone (N = 2–4; fewer at the start), window running across word boundaries |
| | `delta_blick_w2_cross` … `delta_blick_w4_cross` | dependent | dependent | change in the N-phone score from the previous phone (NaN at a sequence's first phone) |
| `blick_within` | `blick_w2_within` … `blick_w4_within`, `delta_blick_w2_within` … `delta_blick_w4_within` | dependent | dependent | same, window reset at each word boundary (delta NaN at a word's first phone) |

> **Disclaimer: stress.** The same all-primary-stress rule is applied to real words, so
> CMUdict's own stress (and any reduced vowels) is ignored. Reduced vowels are scored as
> stressed, which inflates some real-word scores: `THE` (`DH AH1`) scores 10.6. Sentence
> and nonword BLICK values are therefore on the same convention, but real-word values are
> not the BLICK scores you would get with dictionary stress.

## Resources and setup

Two resources, both installed with this package. The phonetic features need nothing extra.

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

Then `linguistic list` should show seven `phonological` metrics.

## Run

```bash
linguistic run -i stimuli.csv -o out/ -m phonological          # all seven
linguistic run -i stimuli.csv -o out/ -m phone:independent     # every metric with a context-independent phone column
linguistic run -i stimuli.csv -o out/ -m phonetic_features -m blick_word
```
