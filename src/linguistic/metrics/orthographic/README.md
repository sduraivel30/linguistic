# Orthographic metrics

Computed from each word's spelling. Sentences and nonwords. Word level, carried to the
stimulus as the mean over its words.

## Context-dependent

| Metric | Columns | Definition (letters w1 … wn) |
| --- | --- | --- |
| `orthographic_ngram` | `orth_bigram_surp`, `orth_trigram_surp` | mean over w1 … wn and the word end of −ln P(letter \| previous 1 or 2 letters) |

Counts are from CMUdict spellings (word types, a–z only), with the word edges marked and
add-one smoothing, so unseen letter sequences are scored instead of NaN. Nats per letter;
higher = less English-like (an ill-formedness score). The mean, not the sum, so it does
not grow with word length.

## Run

```bash
linguistic run -i stimuli.csv -o out/ -m orthographic
```
