"""Regression: ngram_5gram vs the existing langloc 5-gram values.

Reference: ieeg_lanA_correspondence/data/ngram_surprisal_word_level.csv (60 English stimuli;
column word_surprisal_mean_subtoken_nats is in BITS -- lm-zoo get_surprisals prints
-log2 p -- so ours (nats) should equal it x ln 2). Matched on (stimulus_id, word_idx).

Usage (score on a machine with apptainer, e.g. a cluster, then compare where the reference is):
    python scripts/regression/ngram_vs_langloc.py <out dir> --words <words.csv from `-m ngram_5gram`>
Input stimuli: the same file as lm_vs_langloc.py --input-only (English rows are scored).
Exit 1 if any word is unmatched or differs by more than 1e-4 nats.
"""
import argparse
import math
import sys
from pathlib import Path

import pandas as pd

from phonological_vs_langloc import REPOS, _cmp

REF = REPOS / "ieeg_lanA_correspondence" / "data" / "ngram_surprisal_word_level.csv"
TOL = 1e-4


def main(out_dir: str, words_csv: str) -> int:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ref = pd.read_csv(REF, keep_default_na=False, na_values=[""])
    ours = pd.read_csv(words_csv, keep_default_na=False, na_values=[""])
    ours = ours[ours["condition"] == "sentence"]
    m = ref.merge(ours, on=["stimulus_id", "word_idx"], how="outer", indicator=True, suffixes=("_ref", ""))
    print(f"words: reference {len(ref)}, ours {len(ours)}, matched {(m['_merge'] == 'both').sum()}")
    b = m[m["_merge"] == "both"]
    ref_nats = b["word_surprisal_mean_subtoken_nats"] * math.log(2)
    rows = [_cmp("word (lowercase)", b["word"].str.lower(), b["word_ref"].str.lower(), exact=True),
            _cmp("n tokens", b["ngram_n_tokens_5gram"], b["n_subtokens_merged"], exact=True),
            _cmp("surprisal (nats) vs reference x ln2", b["ngram_surprisal_5gram"], ref_nats)]
    res = pd.DataFrame(rows)
    res["n_over_tol"] = [0, 0, int(((b["ngram_surprisal_5gram"] - ref_nats).abs() > TOL).sum())]
    res.to_csv(out / "comparison.csv", index=False)
    print(res.to_string(index=False))
    print(f"OOV words: {int(b['ngram_oov_5gram'].sum())}")
    return int((m["_merge"] != "both").any() or res.loc[:1, "n_differ"].gt(0).any() or res["n_over_tol"].gt(0).any())


if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("out_dir")
    a.add_argument("--words", required=True)
    args = a.parse_args()
    sys.exit(main(args.out_dir, args.words))
