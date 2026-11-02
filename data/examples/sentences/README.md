# Example: sentence list

24 English sentences, 12 words each, run through the pipeline with every registered
metric (`-m all`) so you can see the input and output formats. Regenerate this output
whenever a metric family is added.

## Where it comes from

The sentence condition of the EvLab reading language localizer (`LangLoc_Eng_visual_PTB`;
Fedorenko et al., 2010, *J Neurophysiol* 104:1177-1194).

- Download page: http://www.evlab.mit.edu/resources-all/download-localizer-tasks
  ("Reading" Dropbox folder), fetched 2026-11-02
- File: `langloc_fmri_run1_stim_set1.csv` (rows flagged `S`),
  sha256 `6a545fd8033bcddeaf465a6013f65ed511a2aaf537e493a83cb2515989d8316f`
- No license is stated; EvLab asks that users cite the localizer paper.

## Commands

From the repo root:

```bash
python scripts/make_example_evlab_langloc.py langloc_fmri_run1_stim_set1.csv \
    data/examples/sentences/stimuli.csv sentence
linguistic run -i data/examples/sentences/stimuli.csv -o data/examples/sentences/output -m all
```

The first command keeps the `S` rows and writes the pipeline input. The second prints:

```
skipped elp_frequency: resource elp not installed (`linguistic fetch` or see the family README)
24 stimuli, 288 words, 1039 phones; phone sources {'cmudict': 286, 'missing': 2}; 9 metric(s) -> data/examples/sentences/output
2 word(s) need hand phones: see data/examples/sentences/output/missing_phones.csv
```

Metrics in this output:

- phonological (`phoneme_tp_cross`, `phoneme_tp_within`, `phoneme_tp_backward`, `blick_word`,
  `blick_cross`, `blick_within`): `src/linguistic/metrics/phonological/README.md`
- lexical (`cohort`, `word_duration`, `subtlex_frequency`): `src/linguistic/metrics/lexical/README.md`.
  `elp_frequency` is skipped because the ELP export is not installed. `word_duration_ms`
  and `uniqueness_point_ms` are empty: these stimuli have no hand-label timing.

## Input: `stimuli.csv`

```
stimulus_id,condition,text
sentence_01,sentence,JUST THE BAREST SUGGESTION OF A HEEL IS FOUND ON TEENAGE PUMPS
```

## Output: `output/`

| File | One row per | What it holds |
| --- | --- | --- |
| `phones.csv` | phone | phone label and position, then one column per phone-level metric value; `start`/`end` are empty because these phones come from CMUdict, not hand labels |
| `words.csv` | word | position, CMUdict tag (`in_lexicon`, `lexicon_route`), phones and their source, then each metric (phone-level metrics as the mean over the word's phones) |
| `stimuli.csv` | sentence | `n_words`, then for each metric the mean over the sentence's words and `n_valid_<metric>` (how many words had a value) |
| `missing_phones.csv` | word without phones | words that need hand-extracted phones |
| `manifest.json` | run | input checksum, package versions, lexicon, metric definitions, code commit |

`phones.csv` (selected columns; values in nats):

```
stimulus_id,word_idx,phone_in_word,phone,bigram_surp_cross,bigram_surp_within,bigram_surp_back
sentence_01,1,1,JH,,,4.132
sentence_01,1,2,AH,1.721,1.721,2.29
sentence_01,1,3,S,2.638,2.638,1.613
sentence_01,1,4,T,1.651,1.651,4.97
```

`words.csv` (selected columns): `THE` gets a high cross-word value because its `DH`
follows the `T` of `JUST`:

```
stimulus_id,word_idx,word,phones,bigram_surp_cross,bigram_surp_within,bigram_surp_back
sentence_01,1,JUST,JH AH S T,2.003,2.003,3.251
sentence_01,2,THE,DH AH,6.075,2.745,4.877
sentence_01,3,BAREST,B EH R AH S T,2.387,2.182,2.35
```

`stimuli.csv` (selected columns): `trigram_surp_within` is defined for only 7 of the 12
words, because words with fewer than 3 phones have no within-word trigram:

```
stimulus_id,n_words,bigram_surp_cross,n_valid_bigram_surp_cross,trigram_surp_within,n_valid_trigram_surp_within
sentence_01,12,3.298,12,2.371,7
```

`missing_phones.csv` (the 2 words not in CMUdict):

```
stimulus_id,condition,word_idx,word
sentence_20,sentence,4,CLOTHESBRUSH
sentence_21,sentence,11,MORTARED
```
