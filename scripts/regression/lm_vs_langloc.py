"""Regression: the vendored GPT-2 XL metrics vs the existing langloc outputs.

References (not in this repo):
  Speech_Lang_iEEG/surprisal/linguistic/results/gpt2-xl/word_level.csv
      word_surprisal_mean_subtoken_nats, n_subtokens (Greta's extractor, CUDA float32)
  ieeg_lanA_correspondence/data/gpt2xl_word_level_with_entropy.csv
      word_entropy_nats (a later run; its surprisal agrees with the above within 2.2e-5)
Input: all 120 langloc stimuli, text = the written words GPT-2 scored there
(`stimulus_text`, punctuation already removed). Matched on (stimulus_id, word_idx).

Needs the GPT-2 XL weights: `linguistic fetch gpt2-xl` (see src/linguistic/metrics/lm/README.md).

Usage:
    python scripts/regression/lm_vs_langloc.py <out dir>                 # score here, then compare
    python scripts/regression/lm_vs_langloc.py <out dir> --input-only    # only write <out dir>/stimuli.csv
    python scripts/regression/lm_vs_langloc.py <out dir> --words <words.csv>
        compare a words.csv scored elsewhere (e.g. on ORCD: `linguistic run -i stimuli.csv
        -o <dir> -m lm_gpt2-xl`) without loading the model here
Exit 1 if any value differs by more than 1e-3 nats (float32 results differ slightly
across hardware) or n_subtokens differs at all.
"""
import sys
from pathlib import Path

import pandas as pd

from linguistic.assemble import run_metrics
from linguistic.io import load_stimuli
from linguistic.normalize import Lexicon, build_units
from linguistic.registry import Context, select

from phonological_vs_langloc import REPOS, _cmp

SURP = REPOS / "Speech_Lang_iEEG" / "surprisal" / "linguistic" / "results" / "gpt2-xl" / "word_level.csv"
ENT = REPOS / "ieeg_lanA_correspondence" / "data" / "gpt2xl_word_level_with_entropy.csv"
TOL = 1e-3


def main(out_dir: str, words_csv: str | None = None, input_only: bool = False) -> int:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ref = pd.read_csv(SURP, dtype={"item_id": str}, keep_default_na=False, na_values=[""])   # a nonword is "nan"
    stim = (ref.drop_duplicates("stimulus_id")[["stimulus_id", "condition", "stimulus_text"]]
            .rename(columns={"stimulus_text": "text"}))
    stim["condition"] = stim["condition"].map({"English": "sentence", "Nonsense": "nonword"})
    stim.to_csv(out / "stimuli.csv", index=False)
    if input_only:
        print(f"wrote {out / 'stimuli.csv'} ({len(stim)} stimuli)")
        return 0

    if words_csv:
        words = pd.read_csv(words_csv, keep_default_na=False, na_values=[""])
    else:
        s = load_stimuli(out / "stimuli.csv")
        lex = Lexicon()
        u = build_units(s, lex)
        words = run_metrics(Context(s, u.words, u.phones, lex), select(["lm_gpt2-xl"])).words
        words.to_csv(out / "words.csv", index=False)

    ent = pd.read_csv(ENT)[["stimulus_id", "word_idx", "word_entropy_nats"]]
    m = (words.merge(ref, on=["stimulus_id", "word_idx"], how="outer", indicator=True,
                       suffixes=("", "_ref"))
         .merge(ent, on=["stimulus_id", "word_idx"], how="left"))
    print(f"words: ours {len(words)}, reference {len(ref)}, matched {(m['_merge'] == 'both').sum()}")
    b = m[m["_merge"] == "both"]
    rows = [_cmp("word text", b["word"], b["word_ref"], exact=True),
            _cmp("n_subtokens", b["lm_n_subtokens_gpt2xl"], b["n_subtokens"], exact=True),
            _cmp("mean sub-token surprisal", b["lm_surprisal_gpt2xl"], b["word_surprisal_mean_subtoken_nats"]),
            _cmp("entropy", b["lm_entropy_gpt2xl"], b["word_entropy_nats"])]
    res = pd.DataFrame(rows)                         # _cmp counts differences at 1e-9; recount at TOL
    res["n_over_1e-3"] = [0, 0] + [
        int(((b[a] - b[c]).abs() > TOL).sum()) for a, c in
        (("lm_surprisal_gpt2xl", "word_surprisal_mean_subtoken_nats"), ("lm_entropy_gpt2xl", "word_entropy_nats"))]
    res.to_csv(out / "comparison.csv", index=False)
    print(res.to_string(index=False))
    bad = (m["_merge"] != "both").any() or res.loc[:1, "n_differ"].gt(0).any() or res["n_over_1e-3"].gt(0).any()
    return int(bad)


if __name__ == "__main__":
    import argparse
    a = argparse.ArgumentParser()
    a.add_argument("out_dir")
    a.add_argument("--words")
    a.add_argument("--input-only", action="store_true")
    args = a.parse_args()
    sys.exit(main(args.out_dir, args.words, args.input_only))
