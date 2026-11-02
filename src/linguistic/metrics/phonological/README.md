# Phonological metrics

Computed from each word's phones. Sentences and nonwords. Phone metrics also appear in
`words.csv` as the mean over the word's phones (not across words). Surprisal in nats;
n-gram counts from CMUdict (stress stripped); unseen n-grams are NaN.

## Context-independent

| Metric | Columns | Definition |
| --- | --- | --- |
| `phonetic_features` | `feat_<feature>` | 1 if the phone has the feature, else 0; word value = share of its phones |
| `blick_cross` / `blick_within` | `blick_w1_*` | BLICK score of the phone alone |

Features follow Mesgarani et al. (2014), as in
[this TIMIT feature table](https://github.com/ChangLabUcsf/HamiltonOganianChang2020/blob/master/util1/Timit_features.mat):

| Feature | Phones |
| --- | --- |
| `dorsal` | CH G JH K NG SH |
| `coronal` | CH D JH N S T Z |
| `labial` | B F M P V |
| `high` | EH EY IH IY UH UW |
| `front` | AE EH EY IH IY |
| `low` | AA AE AH AO AW AY |
| `back` | AA AH AO OW UH UW |
| `plosive` | B D G K P T |
| `fricative` | CH DH F HH JH S SH TH V Z ZH |
| `syllabic` | AA AE AH AO AW AY EH ER EY IH IY OW OY UH UW |
| `nasal` | M N NG |
| `voiced` | AA AE AH AO AW AY B D DH EH EY G IH IY L M N NG OW OY R UH UW V W Y Z |
| `obstruent` | B CH D DH F G HH JH K P S SH T TH V Z ZH |
| `sonorant` | AA AE AH AO AW AY EH EY IH IY L M N NG OW OY R UH UW W Y |

Kept as published: `ER`, `JH`, `ZH` are not `voiced`; `ER` is not `sonorant`.

Single-phone BLICK scores the phone as if it were a whole word, so it mostly reflects
whether that phone could stand alone (`T` = 15.6).

## Context-dependent

| Metric | Columns | Definition (sequence … A B C …) |
| --- | --- | --- |
| `phoneme_tp_cross` | `bigram_surp_cross`, `trigram_surp_cross`, `log_bigram_freq_cross` | −ln P(B\|A), −ln P(C\|A,B), ln count(A,B), across word boundaries |
| `phoneme_tp_within` | `bigram_surp_within`, `trigram_surp_within`, `log_bigram_freq_within` | same, reset at each word boundary |
| `phoneme_tp_backward` | `bigram_surp_back`, `trigram_surp_back` | −ln P(B\|C), −ln P(X\|Y,Z), across word boundaries |
| `cohort_phoneme` | `cohort_surprisal`, `cohort_entropy`, `cohort_size` | within the word, over the frequency-weighted cohort (CMUdict words sharing the phonemes so far, SUBTLEX-US counts + 1): −ln P(phoneme \| earlier phonemes of the word), entropy over the remaining words, and their number (Brodbeck et al. 2018; Gwilliams et al. 2022). NaN once no word shares the prefix |
| `blick_word` | `blick_word` | BLICK score of the whole word |
| `blick_cross` / `blick_within` | `blick_w2_*` … `blick_w4_*`, `delta_blick_*` | BLICK score of the last 2–4 phones, and its change from the previous phone |

BLICK: 0 = English-like, higher = less (`blick` 0.12, `bnick` 5.95). All vowels get primary
stress, real words included, so reduced vowels inflate some scores (`THE` = 10.6).

## Setup

```bash
pip install -e .                                          # includes python-BLICK 0.2.12
python -c "import nltk; nltk.download('cmudict')"
linguistic fetch subtlex_us                               # word frequencies for the cohort
```

## Run

```bash
linguistic run -i stimuli.csv -o out/ -m phonological
```
