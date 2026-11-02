"""Frequency-weighted cohort surprisal and entropy per phoneme (within the word).

As in Brodbeck et al. (2018, Curr Biol; 2022, eLife) and Gwilliams et al. (2022, Nat
Commun). For phoneme k of a word, the cohort C_k is every lexicon pronunciation whose first
k phonemes equal the word's; C_0 is the whole lexicon. With F(C) the summed frequency:
  cohort_surprisal = -ln( F(C_k) / F(C_{k-1}) )       nats
  cohort_entropy   = -sum_{v in C_k} p(v) ln p(v),  p(v) = f(v) / F(C_k)   nats
  cohort_size      = |C_k|
Lexicon: every CMUdict pronunciation (stress stripped). f = SUBTLEX-US FREQcount + 1 by
lowercase spelling, split evenly over the word's pronunciations. When the word leaves the
lexicon (no pronunciation shares the prefix: nonwords, or hand phones unlike CMUdict),
surprisal is NaN from that phoneme on and entropy and size are NaN / 0.
"""
from __future__ import annotations

import math
from collections import defaultdict
from functools import lru_cache

import numpy as np

from ...registry import Metric, register
from ..lexical.frequency import _subtlex
from .ngram import _KEYS


@lru_cache(maxsize=2)
def prefix_stats(lexicon) -> dict[tuple[str, ...], tuple[float, float, int]]:
    """prefix -> (F = sum f, G = sum f ln f, N = number of pronunciations)."""
    lexicon._load()
    counts = _subtlex()
    F, G, N = defaultdict(float), defaultdict(float), defaultdict(int)
    for word, prons in lexicon._d.items():
        f = (counts.get(word.lower(), (0, 0.0))[0] + 1) / len(prons)
        flf = f * math.log(f)
        for ph in prons:
            for k in range(0, len(ph) + 1):
                p = ph[:k]
                F[p] += f
                G[p] += flf
                N[p] += 1
    return {p: (F[p], G[p], N[p]) for p in F}


def cohort_values(phones: list[str], stats) -> list[tuple[float, float, int]]:
    out, prev = [], stats[()][0]
    for k in range(1, len(phones) + 1):
        s = stats.get(tuple(phones[:k]))
        if s is None or prev is None:
            out.append((np.nan, np.nan, 0))
            prev = None
            continue
        F, G, N = s
        out.append((-math.log(F / prev), math.log(F) - G / F, N))
        prev = F
    return out


@register
class CohortPhoneme(Metric):
    name = "cohort_phoneme"
    family = "phonological"
    level = "phone"
    requires = frozenset({"phones"})
    resources = ("subtlex_us",)
    context = {"phone": "dependent", "word": "dependent"}
    columns = ("cohort_surprisal", "cohort_entropy", "cohort_size")
    description = ("Frequency-weighted cohort, within the word: surprisal of the phoneme given "
                   "the word's earlier phonemes, entropy over the remaining candidate words, and "
                   "their number (CMUdict, SUBTLEX-US counts + 1).")

    def compute(self, ctx):
        stats = prefix_stats(ctx.lexicon)
        out = ctx.phones[_KEYS].copy()
        vals = np.full((len(out), 3), np.nan)
        ph = ctx.phones.sort_values(["stimulus_id", "word_idx", "phone_in_word"])
        for _, g in ph.groupby(["stimulus_id", "word_idx"], sort=False):
            vals[out.index.get_indexer(g.index)] = cohort_values(g["phone"].tolist(), stats)
        out[list(self.columns)] = vals
        return out
