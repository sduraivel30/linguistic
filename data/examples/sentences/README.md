# Example: sentences

The 20 Harvard sentences of Lists 1 and 2 (IEEE Recommended Practice for Speech Quality
Measurements, *IEEE Trans. Audio Electroacoust.* 17:225–246, 1969, Appendix C; text as in
https://www.cs.columbia.edu/~hgs/audio/harvard.html), written in `stimuli.csv` as published.

```bash
linguistic run -i data/examples/sentences/stimuli.csv -o data/examples/sentences/output -m all
```

Input (`stimuli.csv`):

```
stimulus_id,condition,text
harvard_list1_01,sentence,The birch canoe slid on the smooth planks.
```

Output (`output/`): `phones.csv`, `words.csv` and `stimuli.csv` (one row per phone, word,
sentence), `columns.csv` (what each column is), `missing_phones.csv`, `manifest.json`.

Made on a GPU node with every model installed except ELP. Phones come from CMUdict (every
word is in it), so timing columns are empty.
