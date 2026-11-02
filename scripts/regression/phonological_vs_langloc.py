"""Regression: the vendored phonological metrics vs the existing langloc outputs.

Reference outputs (not in this repo):
  Speech_Lang_iEEG/surprisal/phoneme/tp_blick_per_phone.csv        forward cross/within TP, BLICK, word assignment
  Speech_Lang_iEEG/surprisal/phoneme/plots/backward_values_<cond>.npz   backward bigram/trigram
Input: the canonical MT transcriptions (one Audacity file per item, both tiers), file
choice as in word_form_boundary_analysis mt_phones.collect_mt_files (prefer *_MT_SD.txt),
minus the 6 items excluded in Speech_Lang_iEEG phoneme_data_exclusions.py.

Usage:
    python scripts/regression/phonological_vs_langloc.py <out dir>
Writes <out dir>/stimuli.csv, the pipeline outputs, and comparison.csv; prints a summary.
Phones are matched on (item, start time, phone label). Exit 1 if any value differs by
more than 1e-9 or any phone is unmatched.
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from linguistic.assemble import run_metrics
from linguistic.io import is_phone_label, load_stimuli, read_audacity_labels
from linguistic.normalize import Lexicon, build_units
from linguistic.registry import Context, select

REPOS = Path(__file__).resolve().parents[3]
PHON = REPOS / "Speech_Lang_iEEG" / "surprisal" / "phoneme"
MT_DIR = Path("/Users/dsuseendar/MIT Dropbox/Suseendrakumar Duraivel/Projects/intracranial"
              "/langloc/langloc_audio_stimuli/transcribed_labels_MT")
EXCLUDED = {("Nonsense", 3), ("English", 7), ("English", 24), ("Nonsense", 43),
            ("Nonsense", 45), ("English", 56)}
COND = {"English": "sentence", "Nonsense": "nonword"}
TOL = 1e-9


def mt_files():
    best = {}
    for p in sorted(MT_DIR.glob("*.txt")):
        m = re.match(r"^(\d{3})_(English|Nonsense)_", p.name, re.IGNORECASE)
        if not m or p.name.endswith(".bak"):
            continue
        key = (m.group(1), m.group(2).capitalize())
        if key not in best or ("_MT_SD" in p.name and "_MT_SD" not in best[key].name):
            best[key] = p
    return best


def build_input(out: Path) -> Path:
    rows = []
    for (run, cond), p in sorted(mt_files().items()):
        if (cond, int(run)) in EXCLUDED:
            continue
        words = [r["label"] for r in read_audacity_labels(p)
                 if not is_phone_label(r["label"]) and r["label"].strip().lower() not in ("sil", "sp", "")]
        rows.append({"stimulus_id": f"{run}_{cond}", "condition": COND[cond],
                     "text": " ".join(words), "labels": str(p)})
    path = out / "stimuli.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def main(out_dir: str) -> int:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stim = load_stimuli(build_input(out))
    lex = Lexicon()
    u = build_units(stim, lex)
    t = run_metrics(Context(stim, u.words, u.phones, lex), select(["phonological"]))  # TP + BLICK
    t.phones.to_csv(out / "phones.csv", index=False)
    t.words.to_csv(out / "words.csv", index=False)

    ours = t.phones.copy()
    ours["run"] = ours["stimulus_id"].str[:3].astype(int)
    ours["cond"] = ours["stimulus_id"].str[4:]
    ours["t"] = ours["start"].round(6)

    ref = pd.read_csv(PHON / "tp_blick_per_phone.csv", dtype={"run": int})
    ref = ref[~ref.apply(lambda r: (r["condition"], r["run"]) in EXCLUDED, axis=1)]
    ref["t"] = ref["start"].round(6)
    back = pd.concat([pd.DataFrame({"run": d["run"].astype(int), "cond": c,
                                    "t": np.round(d["start"].astype(float), 6),
                                    "phone": d["phones"],
                                    "bigram_surp_back": d["bigram_back"],
                                    "trigram_surp_back": d["trigram_back"]})
                      for c in ("English", "Nonsense")
                      for d in [np.load(PHON / "plots" / f"backward_values_{c}.npz", allow_pickle=True)]])

    ref["phone_ref"] = ref["phone"]
    m = ours.merge(ref.rename(columns={"condition": "cond"}), on=["run", "cond", "t", "phone"],
                   how="outer", suffixes=("", "_ref"), indicator=True)
    lines = [f"phones: ours {len(ours)}, reference {len(ref)}, matched "
             f"{(m['_merge'] == 'both').sum()}, only ours {(m['_merge'] == 'left_only').sum()}, "
             f"only reference {(m['_merge'] == 'right_only').sum()}"]
    both = m[m["_merge"] == "both"]
    rows, bad = [], (m["_merge"] != "both").any()
    rows.append(_cmp("phone label", both["phone"], both["phone_ref"], exact=True))
    rows.append(_cmp("word_idx", both["word_idx"] - 1, both["word_idx_ref"], exact=True))
    rows.append(_cmp("phone_in_word", both["phone_in_word"] - 1, both["phone_in_word_ref"], exact=True))
    for c in ("bigram_surp_cross", "trigram_surp_cross", "log_bigram_freq_cross",
              "bigram_surp_within", "trigram_surp_within", "log_bigram_freq_within"):
        rows.append(_cmp(c, both[c], both[f"{c}_ref"]))
    blick = [f"blick_w{w}_{s}" for s in ("cross", "within") for w in (1, 2, 3, 4)] + \
            [f"delta_blick_w{w}_{s}" for s in ("cross", "within") for w in (2, 3, 4)]
    for c in blick:
        if c.endswith("_within"):
            # the reference left these NaN for the 349 phones whose word assignment was
            # repaired (BLICK needs the generator); compare where it has a value
            have = both[f"{c}_ref"].notna()
            r = _cmp(c, both.loc[have, c], both.loc[have, f"{c}_ref"])
            r["ref_nan_we_fill"] = int((~have & both[c].notna()).sum())
            rows.append(r)
        else:
            rows.append(_cmp(c, both[c], both[f"{c}_ref"]))

    mb = ours.merge(back, on=["run", "cond", "t", "phone"], how="inner", suffixes=("", "_ref"))
    lines.append(f"backward: reference {len(back)}, matched {len(mb)}")
    bad |= len(mb) != len(back)
    for c in ("bigram_surp_back", "trigram_surp_back"):
        rows.append(_cmp(c, mb[c], mb[f"{c}_ref"]))

    res = pd.DataFrame(rows)
    res.to_csv(out / "comparison.csv", index=False)
    bad |= (res["n_differ"] > 0).any()
    print("\n".join(lines))
    print(res.to_string(index=False))
    return int(bad)


def _cmp(name, a, b, exact=False):
    a, b = pd.Series(a).reset_index(drop=True), pd.Series(b).reset_index(drop=True)
    if exact:
        differ = (a != b)
        return {"column": name, "n": len(a), "n_differ": int(differ.sum()), "max_abs_diff": None,
                "nan_pattern_differs": 0}
    a, b = a.astype(float), b.astype(float)
    nan_diff = a.isna() != b.isna()
    both = a.notna() & b.notna()
    d = (a[both] - b[both]).abs()
    return {"column": name, "n": len(a), "n_differ": int(nan_diff.sum() + (d > TOL).sum()),
            "max_abs_diff": float(d.max()) if len(d) else 0.0,
            "nan_pattern_differs": int(nan_diff.sum())}


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
