# Orthographic metrics

Computed from each word's spelling. Sentences and nonwords. Word level, carried to the
stimulus as the mean over its words.

## Context-dependent

| Metric | Columns | Definition (letters w1 … wn) |
| --- | --- | --- |
| `orthographic_ngram` | `orth_bigram_surp`, `orth_trigram_surp` | mean over w1 … wn and the word end of −ln P(letter \| previous 1 or 2 letters) |
| `orthographic_neighbourhood` | `orth_old20`, `orth_nbhd_entropy_b*`, `orth_nbhd_surprisal_b*` | OLD20; entropy and surprisal of the word's edit-distance neighbourhood (below) |

Counts are from CMUdict spellings (word types, a–z only), with the word edges marked and
add-one smoothing, so unseen letter sequences are scored instead of NaN. Nats per letter;
higher = less English-like (an ill-formedness score). The mean, not the sum, so it does
not grow with word length.

Neighbourhood: every CMUdict word w, weighted by f(w) = SUBTLEX-US count + 1, the word
itself left out. With d = letter edit distance ÷ the longer length,
P(w | x) ∝ f(w) e^(−β d) (after Bailey & Hahn 2001), for β = 10 and 20:

- entropy = −Σ P(w | x) ln P(w | x): low when one neighbour dominates, high when many are
  about equally close;
- surprisal = −ln Σ P_lex(w) e^(−β d), P_lex = f / Σ f: high when the word is far from
  everything frequent;
- OLD20 = mean raw edit distance to the 20 nearest words (Yarkoni et al. 2008).

Without the length normalisation, frequent short words (*a*, *to*) dominate and *xqzt*
looks more word-like than *blick*. The phone version is `phonological_neighbourhood`.
Needs `linguistic fetch subtlex_us`.

## Run

```bash
linguistic run -i stimuli.csv -o out/ -m orthographic
```
