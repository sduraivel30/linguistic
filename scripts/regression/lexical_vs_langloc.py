"""Regression: the vendored lexical metrics vs the existing langloc lexical word-event table.

Reference (not in this repo):
  Speech_Lang_iEEG/neural/results/word_boundary/lexical_word_events.csv
  (build_lexical_word_events.py; one row per word-onset event, word_idx 0-based)
Input: the same 114 MT items as phonological_vs_langloc.py (shared builder).
Words are matched on (item, word_idx, spelling).

Usage:
    python scripts/regression/lexical_vs_langloc.py <out dir>
Exit 1 if any matched value differs by more than 1e-9.
"""
import sys
from pathlib import Path

import pandas as pd

from linguistic.assemble import run_metrics
from linguistic.io import load_stimuli
from linguistic.normalize import Lexicon, build_units
from linguistic.registry import Context, select

from phonological_vs_langloc import EXCLUDED, REPOS, _cmp, build_input

REF = REPOS / "Speech_Lang_iEEG" / "neural" / "results" / "word_boundary" / "lexical_word_events.csv"


def main(out_dir: str) -> int:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stim = load_stimuli(build_input(out))
    lex = Lexicon()
    u = build_units(stim, lex)
    t = run_metrics(Context(stim, u.words, u.phones, lex),
                    select(["cohort", "word_duration", "subtlex_frequency"]))
    t.words.to_csv(out / "words.csv", index=False)

    ours = t.words.copy()
    ours["run"] = ours["stimulus_id"].str[:3].astype(int)
    ours["cond"] = ours["stimulus_id"].str[4:]
    ours["wi"] = ours["word_idx"] - 1
    ours["w"] = ours["word"].str.lower()

    ref = pd.read_csv(REF, dtype={"run": int}, keep_default_na=False, na_values=[""])
    n_ref = len(ref)
    ref = ref[~ref.apply(lambda r: (r["condition"], r["run"]) in EXCLUDED, axis=1)]
    ref = ref.rename(columns={"condition": "cond", "word_idx": "wi"})
    ref["w"] = ref["word"].astype(str).str.lower()

    m = ref.merge(ours, on=["run", "cond", "wi", "w"], how="left", suffixes=("_ref", ""),
                  indicator=True)
    print(f"reference events {n_ref}, in the 114 items {len(ref)}, matched "
          f"{(m['_merge'] == 'both').sum()}, unmatched {(m['_merge'] != 'both').sum()}")
    b = m[m["_merge"] == "both"]
    rows = [_cmp("start (s)", b["onset"], b["start"]),
            _cmp("n_phones", b["n_phones"], b["n_phones_ref"]),
            _cmp("word_duration_ms", b["word_duration_ms"], b["word_duration_ms_ref"]),
            _cmp("cohort_density_log10", b["cohort_density_log10"], b["cohort_density_log10_ref"]),
            _cmp("uniqueness_point_idx", b["uniqueness_point_idx"], b["uniqueness_point_idx_ref"]),
            _cmp("uniqueness_point_ms", b["uniqueness_point_ms"], b["uniqueness_point_ms_ref"]),
            _cmp("up_reached", b["up_reached"], b["up_reached_ref"].astype(float)),
            _cmp("subtlex_lg10wf vs word_freq_log", b["subtlex_lg10wf"], b["word_freq_log"])]
    res = pd.DataFrame(rows)
    res.to_csv(out / "comparison.csv", index=False)
    print(res.to_string(index=False))
    return int((res["n_differ"] > 0).any() or (m["_merge"] != "both").any())


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
