"""Word frequency (word level, sentences only).

SUBTLEX-US Lg10WF vendored from Speech_Lang_iEEG @ c55d0a9,
neural/word_boundary/build_lexical_word_events.py (lowercase spelling lookup, NaN for
nonwords and for words not in the list). New here: a hyphenated word not found as written
is looked up in its closed form ('ice-cream' -> 'icecream'); Zipf; ELP HAL frequency.
"""
from __future__ import annotations

import math
from functools import lru_cache

import pandas as pd

from ... import resources
from ...registry import Metric, register

SUBTLEX_MILLIONS = 51.0        # corpus size: SUBTLWF = FREQcount / 51
SUBTLEX_TYPES_MILLIONS = 0.074286   # 74,286 word types
ELP_WORD_COL, ELP_FREQ_COL = "Word", "Log_Freq_HAL"


def _lookup(table: dict, word: str):
    key = word.lower()
    if key in table:
        return table[key]
    if "-" in key:
        return table.get(key.replace("-", ""))
    return None


@lru_cache(maxsize=1)
def _subtlex() -> dict[str, tuple[int, float]]:
    df = pd.read_csv(resources.path("subtlex_us"), sep="\t", keep_default_na=False,
                     na_values=[""], dtype={"Word": str})
    return {w.lower(): (int(c), float(lg)) for w, c, lg in zip(df["Word"], df["FREQcount"], df["Lg10WF"])}


@lru_cache(maxsize=1)
def _elp() -> dict[str, float]:
    p = resources.path("elp")
    df = pd.read_csv(p, keep_default_na=False, na_values=["", "#", "NULL"], dtype={ELP_WORD_COL: str})
    missing = [c for c in (ELP_WORD_COL, ELP_FREQ_COL) if c not in df.columns]
    if missing:
        raise ValueError(f"{p}: needs columns {ELP_WORD_COL!r} and {ELP_FREQ_COL!r}; "
                         f"has {list(df.columns)}")
    df = df.dropna(subset=[ELP_FREQ_COL])
    return {w.lower(): float(f) for w, f in zip(df[ELP_WORD_COL], df[ELP_FREQ_COL])}


class _Freq(Metric):
    family = "lexical"
    context = {"word": "independent"}
    level = "word"
    conditions = frozenset({"sentence"})
    nan_reason = "word frequency is undefined for nonwords"


@register
class SubtlexFrequency(_Freq):
    name = "subtlex_frequency"
    resources = ("subtlex_us",)
    columns = ("subtlex_lg10wf", "subtlex_zipf")
    description = ("SUBTLEX-US (Brysbaert & New 2009): Lg10WF = log10(FREQcount + 1); "
                   "Zipf = log10((FREQcount + 1) / (51 + 0.074286)) + 3 (van Heuven et al. 2014; "
                   "51 M tokens, 74,286 types). Words not in the list are NaN.")

    def compute(self, ctx):
        t = _subtlex()
        rows = []
        for w in ctx.words.itertuples(index=False):
            hit = _lookup(t, w.word)
            if hit:
                rows.append({"stimulus_id": w.stimulus_id, "word_idx": w.word_idx,
                             "subtlex_lg10wf": hit[1],
                             "subtlex_zipf": math.log10((hit[0] + 1) / (SUBTLEX_MILLIONS + SUBTLEX_TYPES_MILLIONS)) + 3})
        return pd.DataFrame(rows, columns=["stimulus_id", "word_idx", *self.columns])


@register
class ElpFrequency(_Freq):
    name = "elp_frequency"
    resources = ("elp",)
    columns = ("elp_log_freq_hal",)
    description = ("English Lexicon Project (Balota et al. 2007) Log_Freq_HAL, the log HAL-corpus "
                   "frequency, from a user-supplied ELP export. Words not in it are NaN.")

    def compute(self, ctx):
        t = _elp()
        rows = [{"stimulus_id": w.stimulus_id, "word_idx": w.word_idx, "elp_log_freq_hal": v}
                for w in ctx.words.itertuples(index=False) if (v := _lookup(t, w.word)) is not None]
        return pd.DataFrame(rows, columns=["stimulus_id", "word_idx", *self.columns])
