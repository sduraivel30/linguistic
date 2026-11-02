# Example: nonwords

24 nonword lists from the EvLab reading language localizer (Fedorenko et al., 2010), file
`langloc_fmri_run1_stim_set1.csv` (rows `N`) from
http://www.evlab.mit.edu/resources-all/download-localizer-tasks.

```bash
python scripts/make_example_evlab_langloc.py langloc_fmri_run1_stim_set1.csv \
    data/examples/nonwords/stimuli.csv nonword
linguistic run -i data/examples/nonwords/stimuli.csv -o data/examples/nonwords/output -m all
```

Input (`stimuli.csv`):

```
stimulus_id,condition,text
nonword_01,nonword,POME OY REE HOLILY SHOURN NE SLEOMING WHIMP REE RERE OS OFUKE
```

Output (`output/`): `phones.csv`, `words.csv` and `stimuli.csv` (one row per phone, nonword,
list), `columns.csv` (what each column is), `missing_phones.csv`, `manifest.json`.

```
stimulus_id,word_idx,word,phones,bigram_surp_cross,blick_word
nonword_01,1,POME,,,
nonword_01,2,OY,OY,,3.133
nonword_01,3,REE,R IY,3.074,1.05
```

177 of 288 nonwords are not in CMUdict, so they have no phones until hand labels are
given (`missing_phones.csv`). Word frequency is sentence-only. ELP and GPT-2 XL were not
installed when this was made.
