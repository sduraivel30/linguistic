# Example: nonword list

24 nonword lists, 12 nonwords each, run through the pipeline with every registered metric
(`-m all`) so you can see the input and output formats. Regenerate this output whenever a
metric family is added.

## Where it comes from

The nonword condition of the EvLab reading language localizer (`LangLoc_Eng_visual_PTB`;
Fedorenko et al., 2010, *J Neurophysiol* 104:1177-1194).

- Download page: http://www.evlab.mit.edu/resources-all/download-localizer-tasks
  ("Reading" Dropbox folder), fetched 2026-11-02
- File: `langloc_fmri_run1_stim_set1.csv` (rows flagged `N`),
  sha256 `6a545fd8033bcddeaf465a6013f65ed511a2aaf537e493a83cb2515989d8316f`
- No license is stated; EvLab asks that users cite the localizer paper.

## Commands

From the repo root:

```bash
python scripts/make_example_evlab_langloc.py langloc_fmri_run1_stim_set1.csv \
    data/examples/nonwords/stimuli.csv nonword
linguistic run -i data/examples/nonwords/stimuli.csv -o data/examples/nonwords/output -m all
```

The first command keeps the `N` rows and writes the pipeline input. The second prints:

```
skipped elp_frequency: resource elp not installed (`linguistic fetch` or see the family README)
skipped lm_gpt2-xl: resource gpt2-xl not installed (`linguistic fetch` or see the family README)
24 stimuli, 288 words, 227 phones; phone sources {'missing': 177, 'cmudict': 111}; 10 metric(s) -> data/examples/nonwords/output
177 word(s) need hand phones: see data/examples/nonwords/output/missing_phones.csv
```

Metrics in this output:

- phonological (`phonetic_features`, `phoneme_tp_cross`, `phoneme_tp_within`,
  `phoneme_tp_backward`, `blick_word`, `blick_cross`, `blick_within`):
  `src/linguistic/metrics/phonological/README.md`
- lexical (`cohort`, `word_duration`, `subtlex_frequency`): `src/linguistic/metrics/lexical/README.md`.
  `elp_frequency` and `lm_gpt2-xl` are skipped because the ELP export and the GPT-2 XL
  model are not installed. `word_duration_ms`
  and `uniqueness_point_ms` are empty: these stimuli have no hand-label timing. Word
  frequency is sentence-only, so its columns are empty for nonwords.

## Input: `stimuli.csv`

```
stimulus_id,condition,text
nonword_01,nonword,POME OY REE HOLILY SHOURN NE SLEOMING WHIMP REE RERE OS OFUKE
```

## Output: `output/`

| File | One row per | What it holds |
| --- | --- | --- |
| `phones.csv` | phone | only nonwords whose spelling is in CMUdict have phones; one column per phone-level metric value; `start`/`end` are empty (no hand labels) |
| `words.csv` | nonword | position, CMUdict tag (`in_lexicon`, `lexicon_route`), phones and their source, then each metric, context-independent first (phone-level metrics as the mean over the word's phones) |
| `stimuli.csv` | nonword list | `n_words`, then for each metric the mean over the list's nonwords and `n_valid_<metric>` |
| `columns.csv` | output column | which table it is in, its context label (independent / dependent at that level), metric, how it was aggregated, definition |
| `missing_phones.csv` | nonword without phones | nonwords that need hand-extracted phones |
| `manifest.json` | run | input checksum, package versions, lexicon, metric definitions, code commit |

`words.csv` (selected columns): `POME` is not in CMUdict, so it has no phones and no
values. `OY` follows `POME`, so it has no forward cross-word value: a word without phones
breaks the context.

```
stimulus_id,word_idx,word,phones,bigram_surp_cross,bigram_surp_within,bigram_surp_back
nonword_01,1,POME,,,,
nonword_01,2,OY,OY,,,7.742
nonword_01,3,REE,R IY,3.074,1.999,1.711
```

`stimuli.csv` (selected columns): only 4 of the 12 nonwords in this list have phones,
so `n_valid` is small until hand phones are supplied:

```
stimulus_id,n_words,bigram_surp_cross,n_valid_bigram_surp_cross,trigram_surp_within,n_valid_trigram_surp_within
nonword_01,12,2.797,4,,0
```

`missing_phones.csv` (177 of 288 nonwords):

```
stimulus_id,condition,word_idx,word
nonword_01,nonword,1,POME
nonword_01,nonword,4,HOLILY
```

111 of 288 nonword spellings match a CMUdict entry, mostly short forms and names
(`OY`, `CLIVE`, `PITH`). CMUdict pronunciations may not match how a nonword was read
aloud; hand-extracted phones override them when supplied (`labels`, or `words_labels` +
`phones_labels`, in `stimuli.csv`).
