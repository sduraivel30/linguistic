# Example: sentence list

24 English sentences, 12 words each, run through the pipeline so you can see the input and
output formats. No metrics are registered yet, so this is the output of `linguistic prepare`.

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
linguistic prepare -i data/examples/sentences/stimuli.csv -o data/examples/sentences/output
```

The first command keeps the `S` rows and writes the pipeline input. The second prints:

```
24 stimuli, 288 words, 1039 phones; phone sources {'cmudict': 286, 'missing': 2}; 0 metric(s) -> data/examples/sentences/output
2 word(s) need hand phones: see data/examples/sentences/output/missing_phones.csv
```

## Input: `stimuli.csv`

```
stimulus_id,condition,text
sentence_01,sentence,JUST THE BAREST SUGGESTION OF A HEEL IS FOUND ON TEENAGE PUMPS
```

## Output: `output/`

| File | One row per | What it holds |
| --- | --- | --- |
| `words.csv` | word | position, CMUdict tag (`in_lexicon`, `lexicon_route`), phones and their source |
| `phones.csv` | phone | phone label and position; `start`/`end` are empty because these phones come from CMUdict, not hand labels |
| `stimuli.csv` | sentence | `n_words`; per-sentence metric means will be added as columns |
| `missing_phones.csv` | word without phones | words that need hand-extracted phones |
| `manifest.json` | run | input checksum, package versions, lexicon, code commit |

`words.csv`:

```
stimulus_id,condition,word_idx,word,word_key,in_lexicon,lexicon_route,n_pronunciations,phone_source,onset,offset,hand_word_label,phones,n_phones
sentence_01,sentence,1,JUST,just,True,whole,2,cmudict,,,,JH AH S T,4
```

`phones.csv`:

```
stimulus_id,condition,word_idx,phone_in_word,phone,start,end,phone_idx
sentence_01,sentence,1,1,JH,,,1
```

`missing_phones.csv` (the 2 words not in CMUdict):

```
stimulus_id,condition,word_idx,word
sentence_20,sentence,4,CLOTHESBRUSH
sentence_21,sentence,11,MORTARED
```
