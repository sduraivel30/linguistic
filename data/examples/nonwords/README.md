# Example: nonwords

60 nonwords (no real stem, no real suffix) from Beyersmann, Turney, Arrow & Fischer-Baum,
"Complex nonword reading in patients with acquired dyslexia", https://osf.io/4cf5j
(`Materials/Patient_Morph_Project_Stimuli_OSF.xlsx`, sheet `nonwords`, column
`nonstem+nonsuffix`), licence CC0 1.0. Here as 5 lists of 12, in the published order.

```bash
linguistic run -i data/examples/nonwords/stimuli.csv -o data/examples/nonwords/output -m all
```

Input (`stimuli.csv`):

```
stimulus_id,condition,text
nonword_list_1,nonword,arfase treitege ligose brooslude bermose flishnule blouduck lotternule brealrel breistel bulnete roifel
```

Output (`output/`): `phones.csv`, `words.csv` and `stimuli.csv` (one row per phone, nonword,
list), `columns.csv` (what each column is), `missing_phones.csv`, `manifest.json`.

None of these nonwords is in CMUdict, so none has phones until hand labels are given
(`missing_phones.csv`) and the phone-based metrics are empty. The language models score
every nonword; word frequency and the 5-gram are sentence-only. Made on a GPU node with
every model installed except ELP.
