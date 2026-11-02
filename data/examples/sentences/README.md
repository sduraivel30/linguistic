# Example: sentences

24 sentences from the EvLab reading language localizer (Fedorenko et al., 2010), file
`langloc_fmri_run1_stim_set1.csv` (rows `S`) from
http://www.evlab.mit.edu/resources-all/download-localizer-tasks.

```bash
python scripts/make_example_evlab_langloc.py langloc_fmri_run1_stim_set1.csv \
    data/examples/sentences/stimuli.csv sentence
linguistic run -i data/examples/sentences/stimuli.csv -o data/examples/sentences/output -m all
```

Input (`stimuli.csv`):

```
stimulus_id,condition,text
sentence_01,sentence,JUST THE BAREST SUGGESTION OF A HEEL IS FOUND ON TEENAGE PUMPS
```

Output (`output/`): `phones.csv`, `words.csv` and `stimuli.csv` (one row per phone, word,
sentence), `columns.csv` (what each column is), `missing_phones.csv`, `manifest.json`.

```
stimulus_id,word_idx,word,phones,bigram_surp_cross,blick_word,subtlex_zipf
sentence_01,1,JUST,JH AH S T,2.003,1.72,6.676
sentence_01,2,THE,DH AH,6.075,10.612,7.468

word,lm_surprisal_gpt2xl,lm_surprisal_albertxxl,lm_surprisal_xlnetlarge,lm_surprisal_t5large,ngram_surprisal_5gram
JUST,9.24,3.85,6.85,3.13,7.65
THE,4.46,0.07,3.48,1.31,5.02
```

Made on a GPU node with every model installed except ELP. The text is ALL CAPS as in the
source, which the language models score as written. Phones come from CMUdict, so timing
columns are empty. `CLOTHESBRUSH` and `MORTARED` have no phones.
