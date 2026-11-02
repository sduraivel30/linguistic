"""Stimulus-file loading and Audacity label parsing.

Stimulus file (CSV or TSV), one row per stimulus:
    stimulus_id     required, unique
    condition       required, 'sentence' or 'nonword'
    text            required, space-separated units (words or nonwords)
    words_labels    optional, Audacity label file of word intervals
    phones_labels   optional, Audacity label file of phone intervals (ARPAbet)
    audio_path      optional, only for acoustic metrics

Hand phones need both label files: phones are assigned to words by time.
Relative label/audio paths resolve against the stimulus file's folder.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

CONDITIONS = ("sentence", "nonword")
REQUIRED = ("stimulus_id", "condition", "text")
PATH_COLUMNS = ("words_labels", "phones_labels", "audio_path")


class InputError(ValueError):
    """The stimulus file or a label file is malformed."""


def load_stimuli(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    sep = "\t" if path.suffix.lower() in (".tsv", ".tab") else ","
    df = pd.read_csv(path, sep=sep, dtype=str, keep_default_na=False)

    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise InputError(f"{path}: missing required column(s) {missing}")

    df["stimulus_id"] = df["stimulus_id"].str.strip()
    df["condition"] = df["condition"].str.strip().str.lower()
    df["text"] = df["text"].str.strip()

    dup = df["stimulus_id"][df["stimulus_id"].duplicated()].unique().tolist()
    if dup:
        raise InputError(f"{path}: duplicate stimulus_id {dup}")
    bad = sorted(set(df["condition"]) - set(CONDITIONS))
    if bad:
        raise InputError(f"{path}: condition must be one of {CONDITIONS}, got {bad}")
    empty = df.loc[df["text"] == "", "stimulus_id"].tolist()
    if empty:
        raise InputError(f"{path}: empty text for {empty}")

    for col in PATH_COLUMNS:
        if col not in df.columns:
            df[col] = ""
        df[col] = [_resolve(p, path.parent) for p in df[col]]

    has_w = df["words_labels"] != ""
    has_p = df["phones_labels"] != ""
    lonely = df.loc[has_w ^ has_p, "stimulus_id"].tolist()
    if lonely:
        raise InputError(
            f"{path}: words_labels and phones_labels must be given together; "
            f"only one given for {lonely}"
        )
    return df.reset_index(drop=True)


def _resolve(p: str, base: Path) -> str:
    p = p.strip()
    if not p:
        return ""
    q = Path(p).expanduser()
    return str(q if q.is_absolute() else (base / q))


def read_audacity_labels(path: str | Path) -> list[dict]:
    """Audacity label track: one 'start<TAB>end<TAB>label' line per interval.

    Blank lines and Audacity's frequency-range lines (starting with '\\') are skipped.
    """
    path = Path(path)
    if not path.exists():
        raise InputError(f"label file not found: {path}")
    out = []
    for n, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip() or line.startswith("\\"):
            continue
        parts = line.split("\t")
        if len(parts) < 3:
            raise InputError(f"{path}:{n}: expected start<TAB>end<TAB>label")
        try:
            start, end = float(parts[0]), float(parts[1])
        except ValueError as e:
            raise InputError(f"{path}:{n}: non-numeric time") from e
        if end < start:
            raise InputError(f"{path}:{n}: end < start")
        out.append({"start": start, "end": end, "label": parts[2].strip()})
    out.sort(key=lambda r: r["start"])
    return out
