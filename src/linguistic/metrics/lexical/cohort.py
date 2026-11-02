"""Cohort density, uniqueness point and word duration (word level).

Vendored from Speech_Lang_iEEG @ c55d0a9, neural/word_boundary/build_lexical_word_events.py
(cmudict_prefix_words, build_lexical_table). Unchanged definitions; the word's phones are
whatever the pipeline assigned (hand labels, else CMUdict).

Lexicon = every CMUdict pronunciation, stress stripped. The word's own spelling (lowercase)
is excluded from every count, so a nonword is compared against real words only.
"""
from __future__ import annotations

from collections import defaultdict
from functools import lru_cache

import numpy as np
import pandas as pd

from ...registry import Metric, register


@lru_cache(maxsize=4)
def _prefixes(lexicon) -> dict[tuple[str, ...], frozenset[str]]:
    lexicon._load()
    pre = defaultdict(set)
    for word, prons in lexicon._d.items():
        for ph in prons:
            for k in range(1, len(ph) + 1):
                pre[ph[:k]].add(word.lower())
    return {k: frozenset(v) for k, v in pre.items()}


def cohort_values(phones: list[str], word: str, pre) -> dict:
    me = word.lower()
    counts = [len(pre.get(tuple(phones[:k]), frozenset()) - {me}) for k in range(1, len(phones) + 1)]
    up = next((k for k, c in enumerate(counts, 1) if c == 0), None)
    reached = up is not None
    if up is None:
        up = len(phones)          # never unique (prefix / homophone of another word): UP = word offset
    return {"cohort_density_log10": float(np.log10(1 + sum(counts))),
            "uniqueness_point_idx": up, "up_reached": reached}


@register
class Cohort(Metric):
    name = "cohort"
    family = "lexical"
    level = "word"
    requires = frozenset({"phones"})
    columns = ("cohort_density_log10", "uniqueness_point_idx", "uniqueness_point_ms", "up_reached")
    description = ("cohort_density_log10 = log10(1 + sum over the word's phonemes k of the number "
                   "of CMUdict words, other than the word itself, whose first k phonemes equal the "
                   "word's first k). uniqueness_point_idx = first k where that number is 0 (no other "
                   "word shares the prefix); if never, the last phoneme and up_reached = False. "
                   "uniqueness_point_ms = end of that phoneme minus word onset (needs timing, else NaN).")

    def compute(self, ctx):
        pre = _prefixes(ctx.lexicon)
        ph = ctx.phones.sort_values(["stimulus_id", "word_idx", "phone_in_word"])
        ends = {k: g["end"].to_list() for k, g in ph.groupby(["stimulus_id", "word_idx"])}
        rows = []
        for w in ctx.words.itertuples(index=False):
            if not w.phones:
                continue
            v = cohort_values(w.phones.split(), w.word, pre)
            e = ends[(w.stimulus_id, w.word_idx)]
            v["uniqueness_point_ms"] = (e[v["uniqueness_point_idx"] - 1] - w.onset) * 1000.0
            rows.append({"stimulus_id": w.stimulus_id, "word_idx": w.word_idx, **v})
        out = pd.DataFrame(rows, columns=["stimulus_id", "word_idx", *self.columns])
        out["up_reached"] = out["up_reached"].astype(float)     # 1/0 so the stimulus mean is a proportion
        return out


@register
class WordDuration(Metric):
    name = "word_duration"
    family = "lexical"
    level = "word"
    requires = frozenset({"timing"})
    columns = ("word_duration_ms",)
    description = "Word offset minus onset in ms (hand-label timing; word bounds taken from its phones)."

    def compute(self, ctx):
        w = ctx.words[["stimulus_id", "word_idx"]].copy()
        w["word_duration_ms"] = (ctx.words["offset"] - ctx.words["onset"]) * 1000.0
        return w
