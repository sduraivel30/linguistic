# linguistic

Linguistic metrics for sentence and nonword lists. Every metric is reported at three
levels: per **phone**, per **word** (word or nonword) and per **stimulus** (sentence or
nonword list), in `phones.csv`, `words.csv` and `stimuli.csv`.

## Install

```bash
python -m venv .venv && .venv/bin/pip install -e ".[dev,lm]"
linguistic fetch all          # lexicons and models with a public download
```

## Input

CSV with `stimulus_id`, `condition` (`sentence` / `nonword`) and `text`. Optional hand
labels (Audacity format): `words_labels` + `phones_labels`, or one `labels` file with both
tiers.

Phones come from the hand labels, else from CMUdict. Words with neither are listed in
`missing_phones.csv` and their phone-based metrics are NaN.

## Run

```bash
linguistic list                                   # available metrics
linguistic run -i stimuli.csv -o out/ -m all      # or a family, a metric, or level:context
linguistic represent -i stimuli.csv -o reps/ -m gpt2-xl     # layer activations, embeddings
```

`columns.csv` in each output lists every column with its level and context.
Examples: [`data/examples/`](data/examples/).

## What is computed at each level

A metric is defined at one level and carried up one level: a phone value to the word (mean
over the word's phones, never across words), a word value to the stimulus (mean over its
words). Context-independent (I) means the value depends only on that unit;
context-dependent (D) means it depends on a sequence of phones or words.

| Metric | Phone | Word | Stimulus |
| --- | --- | --- | --- |
| phonetic features (`feat_*`) | I: 1/0 per feature | I: share of phones | – |
| single-phone BLICK (`blick_w1_*`) | I: score | I: mean over phones | – |
| phoneme transitional probability (`*_surp_cross/_within/_back`) | D: surprisal | D: mean over phones | – |
| BLICK windows (`blick_w2-4_*`, `delta_*`) | D: score | D: mean over phones | – |
| cohort surprisal, entropy (`cohort_*`) | D: value | D: mean over phones | – |
| BLICK whole word (`blick_word`) | – | D: score | D: mean over words |
| cohort density, uniqueness point | – | D: value | D: mean over words |
| word frequency (SUBTLEX, ELP) | – | I: value | I: mean over words |
| word duration | – | I: value | I: mean over words |
| LM surprisal, entropy (`lm_*`) | – | D: value | D: mean over words |
| 5-gram surprisal (`ngram_*`, sentences) | – | D: value | D: mean over words |
| LM activations (`represent`) | – | D: per layer | D: last word, mean over words |
| LM input embeddings (`represent`) | – | I: vector | – |
| GloVe embeddings (`represent -m glove-840b`) | – | I: vector | I: last word, mean over words |

Definitions and setup per family:
[phonological](src/linguistic/metrics/phonological/README.md) ·
[lexical](src/linguistic/metrics/lexical/README.md) ·
[lm](src/linguistic/metrics/lm/README.md).

Running on a SLURM cluster: [`slurm/README.md`](slurm/README.md).

## Known limitations

- English only; phones are ARPAbet. No grapheme-to-phoneme prediction: words outside
  CMUdict need hand labels.
- ELP frequency needs an ELP export placed by hand.
- BLICK gives every vowel primary stress, real words included.
- Text is scored as written: case matters for the language models.
- XLNet reads each stimulus after a fixed padding passage.
- The 5-gram model needs Apptainer, and scores sentences only.
- The language models need a GPU for large stimulus sets.

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md). Licence: MIT.

## Tests

```bash
python -m pytest
```

Language-model tests use small test models from the local Hugging Face cache and are
skipped when those are not downloaded.

## References

**Lexicons and data**
- CMU Pronouncing Dictionary. http://www.speech.cs.cmu.edu/cgi-bin/cmudict
- Brysbaert, M., & New, B. (2009). Moving beyond Kučera and Francis: SUBTLEX-US. *Behavior Research Methods*, 41, 977–990.
- van Heuven, W. J. B., Mandera, P., Keuleers, E., & Brysbaert, M. (2014). SUBTLEX-UK (Zipf scale). *Quarterly Journal of Experimental Psychology*, 67, 1176–1190.
- Balota, D. A., et al. (2007). The English Lexicon Project. *Behavior Research Methods*, 39, 445–459.
- Pennington, J., Socher, R., & Manning, C. D. (2014). GloVe: Global vectors for word representation. *EMNLP*.
- IEEE (1969). IEEE recommended practice for speech quality measurements (Harvard sentences). *IEEE Trans. Audio Electroacoust.*, 17, 225–246.
- Beyersmann, E., Turney, A., Arrow, T., & Fischer-Baum, S. Complex nonword reading in patients with acquired dyslexia. OSF, https://osf.io/4cf5j (CC0).

**Methods**
- Mesgarani, N., Cheung, C., Johnson, K., & Chang, E. F. (2014). Phonetic feature encoding in human superior temporal gyrus. *Science*, 343, 1006–1010. Feature table: https://github.com/ChangLabUcsf/HamiltonOganianChang2020
- Hayes, B. BLICK, a phonotactic probability calculator. http://www.linguistics.ucla.edu/people/hayes/BLICK/
- Brodbeck, C., Hong, L. E., & Simon, J. Z. (2018). Rapid transformation from auditory to linguistic representations of continuous speech. *Current Biology*, 28, 3976–3983.
- Gwilliams, L., King, J.-R., Marantz, A., & Poeppel, D. (2022). Neural dynamics of phoneme sequences reveal position-invariant code for content and order. *Nature Communications*, 13, 6606.
- Salazar, J., Liang, D., Nguyen, T. Q., & Kirchhoff, K. (2020). Masked language model scoring. *ACL*.
- Kauf, C., & Ivanova, A. (2023). A better way to do masked language model scoring. *ACL*.
- Schrimpf, M., et al. (2021). The neural architecture of language: Integrative modeling converges on predictive processing. *PNAS*, 118, e2105646118. Code: https://github.com/mschrimpf/neural-nlp

**Models**
- Radford, A., et al. (2019). Language models are unsupervised multitask learners (GPT-2). OpenAI.
- Yang, Z., et al. (2019). XLNet: Generalized autoregressive pretraining for language understanding. *NeurIPS*.
- Lan, Z., et al. (2020). ALBERT: A lite BERT for self-supervised learning of language representations. *ICLR*.
- Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of deep bidirectional transformers for language understanding. *NAACL*.
- Raffel, C., et al. (2020). Exploring the limits of transfer learning with a unified text-to-text transformer (T5). *JMLR*, 21, 1–67.
- Stolcke, A. (2002). SRILM: An extensible language modeling toolkit. *ICSLP*. Run through lm-zoo (Gauthier, J., Hu, J., Wilcox, E., Qian, P., & Levy, R. (2020). SyntaxGym. *ACL System Demonstrations*), https://github.com/cpllab/lm-zoo

**Software**
- Wolf, T., et al. (2020). Transformers: State-of-the-art natural language processing. *EMNLP System Demonstrations*. https://github.com/huggingface/transformers
- Paszke, A., et al. (2019). PyTorch. *NeurIPS*.
- Bird, S., Klein, E., & Loper, E. (2009). *Natural Language Processing with Python* (NLTK). O'Reilly.
- python-BLICK (McAuliffe & Hayes). https://github.com/mmcauliffe/python-BLICK
- NumPy, pandas, huggingface_hub, Typer, Apptainer.

## Citation

If you use this work, please cite
```
@software{Duraivel2026linguistic,
	author = {Duraivel, Kumar},
	title = {linguistic: linguistic metrics for sentence and nonword lists},
	year = {2026},
	version = {0.1.0},
	url = {https://github.com/sduraivel30/linguistic}
}
```
