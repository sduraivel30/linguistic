"""Convert one EvLab reading-localizer stimulus CSV into this package's input format.

Source: EvLab "LangLoc_Eng_visual_PTB" reading localizer (sentences vs nonword
sequences; Fedorenko et al. 2010, J Neurophysiol), public download linked from
http://www.evlab.mit.edu/resources-all/download-localizer-tasks
Each source row: an item number, 12 words (stim2..stim13), condition flag S/N (stim14).

Usage:
    python scripts/make_example_evlab_langloc.py <langloc_fmri_runX_stim_setY.csv> <out stimuli.csv>
"""
import sys
from pathlib import Path

import pandas as pd

CONDITION = {"S": "sentence", "N": "nonword"}


def main(src: str, out: str) -> None:
    raw = pd.read_csv(src, dtype=str, keep_default_na=False)
    words = [f"stim{i}" for i in range(2, 14)]
    tag = Path(src).stem.replace("langloc_fmri_", "")
    rows = []
    for r in raw.itertuples(index=False):
        d = r._asdict()
        flag = d["stim14"].strip()
        rows.append({
            "stimulus_id": f"{tag}_{flag}{int(d['stim1']):02d}",
            "condition": CONDITION[flag],
            "text": " ".join(d[w].strip() for w in words),
        })
    df = pd.DataFrame(rows)
    assert df["stimulus_id"].is_unique
    assert (df["text"].str.split().str.len() == 12).all()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"{len(df)} stimuli {df['condition'].value_counts().to_dict()} -> {out}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
