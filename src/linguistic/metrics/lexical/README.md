# Lexical metrics

Word-level metrics. In `stimuli.csv` each is the mean over the stimulus's words.

**Context:** all lexical metrics are context-independent at the word and stimulus level:
they depend on the word itself (its spelling, phones or timing) and a reference lexicon,
never on the neighbouring words.

## Metrics

| Metric (`-m` name) | Columns | Conditions | Needs | Definition |
| --- | --- | --- | --- | --- |
| `cohort` | `cohort_density_log10` | sentence, nonword | phones | log10(1 + Σ_k n_k), where n_k = number of CMUdict words, other than the word itself, whose first k phonemes equal the word's first k |
| | `uniqueness_point_idx` | | | first k with n_k = 0 (no other word shares the prefix); if that never happens, the last phoneme |
| | `up_reached` | | | 1 if such a k exists, 0 if the word never becomes unique (it is a prefix or homophone of another word); the stimulus mean is the proportion reached |
| | `uniqueness_point_ms` | | phones + timing | end of phoneme `uniqueness_point_idx` minus word onset, ms (NaN without hand-label timing) |
| `word_duration` | `word_duration_ms` | sentence, nonword | timing | word offset minus onset, ms; word bounds come from its phones |
| `subtlex_frequency` | `subtlex_lg10wf` | sentence | SUBTLEX-US | Lg10WF = log10(FREQcount + 1) |
| | `subtlex_zipf` | | | log10((FREQcount + 1) / (51 + 0.074286)) + 3 (van Heuven et al. 2014; corpus 51 M tokens, 74,286 types) |
| `elp_frequency` | `elp_log_freq_hal` | sentence | ELP export | ELP `Log_Freq_HAL`, the log frequency in the HAL corpus |

- Cohort lexicon: every CMUdict pronunciation, stress stripped. A nonword is compared
  against real words, so its density counts real words sharing each prefix, and its
  uniqueness point is where it diverges from every real word.
- Frequencies are looked up by lowercase spelling. A hyphenated word not found as written
  is looked up closed (`ice-cream` → `icecream`), never as a mean of its parts. Words not
  in the list are NaN. Nonwords are NaN by design.

## Resources and setup

Resources live in `$LINGUISTIC_RESOURCES` (default `~/.cache/linguistic`), one folder each.
`linguistic run -m all` skips a metric whose resource is missing and says so. Naming the
metric (`-m elp_frequency`) makes the run fail with the path it expected.

**CMUdict**: see `src/linguistic/metrics/phonological/README.md` (nltk download).

**SUBTLEX-US** (Brysbaert & New 2009, *Behavior Research Methods* 41:977–990):

1. Run `linguistic fetch subtlex_us`. It downloads
   https://www.ugent.be/pp/experimentele-psychologie/en/research/documents/subtlexus/subtlexus2.zip,
   checks the zip md5 (`7644415a9f78270c7c3b6b4b6395967c`), extracts
   `SUBTLEXus74286wordstextversion.txt` to `~/.cache/linguistic/subtlex_us/` and checks its
   md5 (`ee7be8db21f26e1925ac0ea3ab598f32`).
2. By hand instead: download that zip, extract the txt into
   `$LINGUISTIC_RESOURCES/subtlex_us/`.
3. Check: `linguistic run -i <stimuli.csv> -o <out> -m subtlex_frequency` should not print
   "skipped".

**English Lexicon Project** (Balota et al. 2007, *Behavior Research Methods* 39:445–459),
https://elexicon.wustl.edu/

> The ELP site could not be reached on 2026-11-02 (its TLS certificate had expired), so the
> steps below are general and have not been checked against the current site. Only the
> file format in step 3 is required by this code.

1. On the ELP site, open the word query ("Generate"/"Query" for words).
2. Submit your word list (or all words) and select at least the `Word` and
   `Log_Freq_HAL` variables. Download the result as CSV or text.
3. Save it as `$LINGUISTIC_RESOURCES/elp/elp_words.csv`: a comma-separated file whose
   header has a `Word` column and a `Log_Freq_HAL` column (other columns are ignored;
   `#`, `NULL` or empty = missing). If your export names them differently, rename the
   header.
4. Check: `linguistic run -i <stimuli.csv> -o <out> -m elp_frequency`.

## Run

```bash
linguistic fetch subtlex_us                                     # once
linguistic run -i stimuli.csv -o out/ -m lexical                # all four (ELP skipped if not installed)
linguistic run -i stimuli.csv -o out/ -m cohort -m subtlex_frequency
```
