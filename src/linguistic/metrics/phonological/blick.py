"""BLICK phonotactic scores (Hayes; python-BLICK 0.2.12).

Vendored from word_form_boundary_analysis @ 33e104f:
  src/tp_analysis/blick_analysis.py   phones_to_blick (primary stress '1' on every vowel)
  scripts/run_tp_blick_analysis.py    compute_blick_within / compute_blick_cross
Changes from the source: a word without phones breaks the cross-word window (as in
ngram.py), scores are cached per phone string, and `blick_word` (whole-word score)
is new.

BLICK returns a penalty: the weighted sum of phonotactic constraint violations
(0 = fully English-like, higher = less English-like). BLICK errors give NaN.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

from ...registry import Metric, register
from .ngram import _KEYS, _sequences

VOWELS = {"AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY",
          "IH", "IY", "OW", "OY", "UH", "UW"}
WINDOWS = (1, 2, 3, 4)
STRESS = "1"


def blick_string(phones) -> str:
    """ARPAbet phones -> BLICK input, primary stress on every vowel (stress of nonwords is undefined)."""
    return " ".join(p + STRESS if p in VOWELS else p for p in phones)


@lru_cache(maxsize=1)
def _loader():
    try:
        from blick import BlickLoader
    except ImportError as e:
        raise ImportError("BLICK metrics need python-BLICK: pip install python-BLICK==0.2.12 "
                          "(see src/linguistic/metrics/phonological/README.md)") from e
    return BlickLoader()


@lru_cache(maxsize=None)
def score(phones: tuple[str, ...]) -> float:
    if not phones:
        return float("nan")
    try:
        return float(_loader().assessWord(blick_string(phones)))
    except Exception:
        return float("nan")


def _windowed(ctx, within: bool, suffix: str):
    cols = [f"blick_w{w}_{suffix}" for w in WINDOWS] + [f"delta_blick_w{w}_{suffix}" for w in WINDOWS[1:]]
    out = ctx.phones[_KEYS].copy()
    for c in cols:
        out[c] = np.nan
    for idx, seq in _sequences(ctx, within):
        vals = {c: [] for c in cols}
        prev = {}
        for k in range(len(seq)):
            cur = {w: score(tuple(seq[max(0, k - w + 1): k + 1])) for w in WINDOWS}
            for w in WINDOWS:
                vals[f"blick_w{w}_{suffix}"].append(cur[w])
            for w in WINDOWS[1:]:
                vals[f"delta_blick_w{w}_{suffix}"].append(np.nan if k == 0 else cur[w] - prev[w])
            prev = cur
        for c in cols:
            out.loc[idx, c] = vals[c]
    return out


class _Base(Metric):
    family = "phonological"
    requires = frozenset({"phones"})


@register
class BlickCross(_Base):
    name = "blick_cross"
    level = "phone"
    columns = tuple(f"blick_w{w}_cross" for w in WINDOWS) + tuple(f"delta_blick_w{w}_cross" for w in WINDOWS[1:])
    description = ("BLICK score of the last N phones ending at each phone (N = 1-4), window "
                   "running across word boundaries; delta_wN = change from the previous phone.")

    def compute(self, ctx):
        return _windowed(ctx, within=False, suffix="cross")


@register
class BlickWithin(_Base):
    name = "blick_within"
    level = "phone"
    columns = tuple(f"blick_w{w}_within" for w in WINDOWS) + tuple(f"delta_blick_w{w}_within" for w in WINDOWS[1:])
    description = ("BLICK score of the last N phones ending at each phone (N = 1-4), window "
                   "reset at each word boundary; delta_wN is NaN at the word's first phone.")

    def compute(self, ctx):
        return _windowed(ctx, within=True, suffix="within")


@register
class BlickWord(_Base):
    name = "blick_word"
    level = "word"
    columns = ("blick_word",)
    description = "BLICK score of the whole word's phone string."

    def compute(self, ctx):
        w = ctx.words[["stimulus_id", "word_idx", "phones"]].copy()
        w["blick_word"] = [score(tuple(p.split())) if p else np.nan for p in w["phones"]]
        return w[["stimulus_id", "word_idx", "blick_word"]]
