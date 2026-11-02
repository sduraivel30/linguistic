# Example: nonword list

24 nonword lists, 12 nonwords each, run through the pipeline so you can see the input and
output formats. No metrics are registered yet, so this is the output of `linguistic prepare`.

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
linguistic prepare -i data/examples/nonwords/stimuli.csv -o data/examples/nonwords/output
```

The first command keeps the `N` rows and writes the pipeline input. The second prints:

```
24 stimuli, 288 words, 227 phones; phone sources {'missing': 177, 'cmudict': 111}; 0 metric(s) -> data/examples/nonwords/output
177 word(s) need hand phones: see data/examples/nonwords/output/missing_phones.csv
```

## Input: `stimuli.csv`

```
stimulus_id,condition,text
nonword_01,nonword,POME OY REE HOLILY SHOURN NE SLEOMING WHIMP REE RERE OS OFUKE
```

## Output: `output/`

| File | One row per | What it holds |
| --- | --- | --- |
| `words.csv` | nonword | position, CMUdict tag (`in_lexicon`, `lexicon_route`), phones and their source |
| `phones.csv` | phone | only nonwords whose spelling is in CMUdict have phones; `start`/`end` are empty (no hand labels) |
| `stimuli.csv` | nonword list | `n_words`; per-list metric means will be added as columns |
| `missing_phones.csv` | nonword without phones | nonwords that need hand-extracted phones |
| `manifest.json` | run | input checksum, package versions, lexicon, code commit |

`words.csv`: `POME` is not in CMUdict, so it has no phones; `OY` and `REE` are, so they
get CMUdict phones:

```
stimulus_id,condition,word_idx,word,word_key,in_lexicon,lexicon_route,n_pronunciations,phone_source,onset,offset,hand_word_label,phones,n_phones
nonword_01,nonword,1,POME,pome,False,,0,,,,,,0
nonword_01,nonword,2,OY,oy,True,whole,1,cmudict,,,,OY,1
nonword_01,nonword,3,REE,ree,True,whole,1,cmudict,,,,R IY,2
```

`missing_phones.csv` (177 of 288 nonwords):

```
stimulus_id,condition,word_idx,word
nonword_01,nonword,1,POME
nonword_01,nonword,4,HOLILY
```

111 of 288 nonword spellings match a CMUdict entry, mostly short forms and names
(`OY`, `CLIVE`, `PITH`). CMUdict pronunciations may not match how a nonword was read
aloud; hand-extracted phones override them when supplied (`words_labels` +
`phones_labels` columns in `stimuli.csv`).
