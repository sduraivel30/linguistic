"""Run metrics and assemble the word, phone and stimulus tables.

Levels:
  phone table    -- phone-level metric values, one row per phone
  word table     -- word-level metric values; a phone-level metric appears here as
                    the mean over that word's phones
  stimulus table -- every numeric word-table metric averaged over the words of the
                    stimulus (NaN words skipped), with n_valid_<column>; phone-level
                    metrics are therefore a mean of per-word means (each word counts once)

The runner enforces each metric's declared validity: values are set to NaN for
conditions the metric excludes, for words without phones when it requires
phones, and for units without timing when it requires timing.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .registry import LEVELS, Context, Metric


@dataclass
class Tables:
    words: pd.DataFrame
    phones: pd.DataFrame
    stimuli: pd.DataFrame
    report: list[dict] = field(default_factory=list)


def run_metrics(ctx: Context, metrics: list[Metric]) -> Tables:
    words, phones = ctx.words.copy(), ctx.phones.copy()
    taken = set(words.columns) | set(phones.columns)
    word_cols, report = [], []

    for m in metrics:
        clash = set(m.columns) & taken
        if clash:
            raise ValueError(f"{m.name}: column(s) {sorted(clash)} already exist")
        taken |= set(m.columns)

        keys = list(LEVELS[m.level])
        base = words if m.level == "word" else phones
        out = m.compute(ctx)
        _check_output(m, out, keys)
        merged = base[keys].merge(out[keys + list(m.columns)], on=keys, how="left",
                                  validate="one_to_one")
        values = merged[list(m.columns)].copy()
        values.loc[_invalid(m, base).to_numpy(), :] = np.nan
        for c in m.columns:
            base[c] = values[c].to_numpy()

        if m.level == "phone":
            per_word = phones.groupby(["stimulus_id", "word_idx"])[list(m.columns)].mean()
            words = words.merge(per_word.reset_index(), on=["stimulus_id", "word_idx"],
                                how="left")
        word_cols += list(m.columns)
        report.append(_report(m, words))

    stimuli = stimulus_table(ctx.stimuli, words, word_cols)
    return Tables(words=words, phones=phones, stimuli=stimuli, report=report)


def stimulus_table(stimuli: pd.DataFrame, words: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    base = stimuli[["stimulus_id", "condition"]].copy()
    base["n_words"] = base["stimulus_id"].map(words.groupby("stimulus_id").size()).fillna(0).astype(int)
    numeric = [c for c in cols if pd.api.types.is_numeric_dtype(words[c])]
    if not numeric:
        return base
    g = words.groupby("stimulus_id")[numeric]
    means, counts = g.mean(), g.count().add_prefix("n_valid_")
    agg = means.join(counts).reset_index()
    order = ["stimulus_id"] + [x for c in numeric for x in (c, f"n_valid_{c}")]
    return base.merge(agg[order], on="stimulus_id", how="left")


def _check_output(m: Metric, out: pd.DataFrame, keys: list[str]) -> None:
    missing = [c for c in keys + list(m.columns) if c not in out.columns]
    if missing:
        raise ValueError(f"{m.name}: compute() output lacks column(s) {missing}")
    if out.duplicated(keys).any():
        raise ValueError(f"{m.name}: compute() output has duplicate {keys} rows")


def _invalid(m: Metric, base: pd.DataFrame) -> pd.Series:
    bad = ~base["condition"].isin(m.conditions)
    if "phones" in m.requires and m.level == "word":
        bad |= base["phone_source"].isna()
    if "timing" in m.requires:
        bad |= base["onset" if m.level == "word" else "start"].isna()
    return bad


def _report(m: Metric, words: pd.DataFrame) -> dict:
    r = {"metric": m.name, "level": m.level}
    for cond, g in words.groupby("condition"):
        r[f"n_valid_{cond}"] = {c: int(g[c].notna().sum()) for c in m.columns}
        r[f"n_words_{cond}"] = int(len(g))
    return r
