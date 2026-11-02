# Example: EvLab reading language localizer, run 1 / set 1

Shows what goes in and what comes out. No metrics are registered yet, so this is the
output of `linguistic prepare`: word and phone tables, a stimulus table (only `n_words`
until metrics exist), the list of words still needing hand phones, and the manifest.

## Source

EvLab `LangLoc_Eng_visual_PTB` reading localizer: 12-word sentences and 12-nonword
sequences (Fedorenko et al., 2010, *J Neurophysiol* 104:1177-1194). Public download
linked from http://www.evlab.mit.edu/resources-all/download-localizer-tasks
(Dropbox folder "Reading", fetched 2026-11-02). No license is stated; EvLab asks that
uses cite the localizer paper. `source_langloc_fmri_run1_stim_set1.csv` is the file
from that folder, unchanged.

## Files

| File | What it is |
| --- | --- |
| `source_langloc_fmri_run1_stim_set1.csv` | original EvLab file (item, 12 words, S/N flag) |
| `stimuli.csv` | the pipeline input: `stimulus_id`, `condition`, `text` |
| `output/words.csv` | one row per word/nonword: lexicon tag and route, phones and their source |
| `output/phones.csv` | one row per phone (no timing: phones come from CMUdict, not hand labels) |
| `output/stimuli.csv` | one row per stimulus; metric means will be added here |
| `output/missing_phones.csv` | words with no phones yet (need hand-extracted labels) |
| `output/manifest.json` | input checksum, package versions, lexicon, code commit |

## Reproduce

```bash
python scripts/make_example_evlab_langloc.py <downloaded langloc_fmri_run1_stim_set1.csv> \
    data/examples/evlab_langloc_run1_set1/stimuli.csv
linguistic prepare -i data/examples/evlab_langloc_run1_set1/stimuli.csv \
    -o data/examples/evlab_langloc_run1_set1/output
```

## What this run shows

- 48 stimuli (24 sentences, 24 nonword lists), 576 words.
- Sentences: 286 of 288 words are in CMUdict; `CLOTHESBRUSH` and `MORTARED` are not.
- Nonwords: 111 of 288 spellings match a CMUdict entry (short forms and names such as
  `CLIVE`, `PITH`, `OY`) and get CMUdict phones; the other 177 wait for hand phones.
