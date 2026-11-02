"""Phoneme n-gram transitional-probability metrics (phone level).

Vendored from:
  word_form_boundary_analysis @ 33e104f
    src/tp_analysis/corpus.py            NgramModel (CMUdict bigram/trigram counts)
    scripts/run_tp_blick_analysis.py     compute_tp_within / compute_tp_cross
  Speech_Lang_iEEG @ 1ec9f82
    surprisal/phoneme/build_backward_values.py   backward (C -> B) surprisal
Changes from the source: counts are kept in ARPAbet instead of IPA (the source's
ARPAbet->IPA map is one-to-one, so every count and value is identical), and a word
without phones breaks the cross-word context instead of being skipped over.

Counts: every CMUdict entry (all pronunciations), stress stripped, within-word n-grams.
All logs are natural logs; surprisal is in nats. Unseen n-grams give NaN.
"""
from __future__ import annotations

import math
from collections import Counter

import numpy as np
import pandas as pd

from ...registry import Metric, register

_KEYS = ["stimulus_id", "word_idx", "phone_in_word"]


class PhonemeCounts:
    def __init__(self, lexicon):
        lexicon._load()
        self.uni, self.bi, self.tri = Counter(), Counter(), Counter()
        for prons in lexicon._d.values():
            for ph in prons:
                for p in ph:
                    self.uni[p] += 1
                for i in range(len(ph) - 1):
                    self.bi[(ph[i], ph[i + 1])] += 1
                for i in range(len(ph) - 2):
                    self.tri[(ph[i], ph[i + 1], ph[i + 2])] += 1

    @staticmethod
    def _neglog(num: int, den: int) -> float:
        return -math.log(num / den) if num and den else float("nan")

    def fwd_bigram(self, a, b):        # -ln P(b | a)
        return self._neglog(self.bi[(a, b)], self.uni[a])

    def fwd_trigram(self, a, b, c):    # -ln P(c | a, b)
        return self._neglog(self.tri[(a, b, c)], self.bi[(a, b)])

    def back_bigram(self, b, c):       # -ln P(b | c), assigned to b
        return self._neglog(self.bi[(b, c)], self.uni[c])

    def back_trigram(self, x, y, z):   # -ln P(x | y, z), assigned to x
        return self._neglog(self.tri[(x, y, z)], self.bi[(y, z)])

    def log_bigram_freq(self, a, b):
        c = self.bi[(a, b)]
        return math.log(c) if c else float("nan")


_COUNTS: dict[int, PhonemeCounts] = {}


def _counts(ctx) -> PhonemeCounts:
    key = id(ctx.lexicon)
    if key not in _COUNTS:
        _COUNTS[key] = PhonemeCounts(ctx.lexicon)
    return _COUNTS[key]


def _sequences(ctx, within: bool):
    """Yield (row indices into ctx.phones, phone list) per scoring sequence.

    within=True : one sequence per word.
    within=False: one per run of consecutive words that all have phones, so a
                  word without phones breaks the context (no false adjacency).
    """
    ph = ctx.phones.sort_values(["stimulus_id", "word_idx", "phone_in_word"])
    has = ctx.words.set_index(["stimulus_id", "word_idx"])["n_phones"] > 0
    for sid, g in ph.groupby("stimulus_id", sort=False):
        if within:
            for _, w in g.groupby("word_idx", sort=False):
                yield w.index.to_list(), w["phone"].to_list()
            continue
        run_idx, run_ph, prev = [], [], None
        for wi, w in g.groupby("word_idx", sort=False):
            gap = prev is not None and any(not has.get((sid, k), False) for k in range(prev + 1, wi))
            if gap and run_idx:
                yield run_idx, run_ph
                run_idx, run_ph = [], []
            run_idx += w.index.to_list()
            run_ph += w["phone"].to_list()
            prev = wi
        if run_idx:
            yield run_idx, run_ph


def _score(ctx, within: bool, fns: dict) -> pd.DataFrame:
    out = ctx.phones[_KEYS].copy()
    for col in fns:
        out[col] = np.nan
    for idx, seq in _sequences(ctx, within):
        n = len(seq)
        for col, (fn, need_prev, need_next) in fns.items():
            vals = []
            for k in range(n):
                if k - need_prev < 0 or k + need_next >= n:
                    vals.append(np.nan)
                else:
                    vals.append(fn(seq, k))
            out.loc[idx, col] = vals
    return out


class _Base(Metric):
    family = "phonological"
    level = "phone"
    requires = frozenset({"phones"})


@register
class PhonemeTPCross(_Base):
    name = "phoneme_tp_cross"
    columns = ("bigram_surp_cross", "trigram_surp_cross", "log_bigram_freq_cross")
    description = ("Forward phoneme transitional probability over the stimulus's phone "
                   "sequence, context carried across word boundaries. bigram: -ln P(B|A); "
                   "trigram: -ln P(C|A,B); log_bigram_freq: ln count(A,B). CMUdict counts.")

    def compute(self, ctx):
        c = _counts(ctx)
        return _score(ctx, within=False, fns={
            "bigram_surp_cross": (lambda s, k: c.fwd_bigram(s[k - 1], s[k]), 1, 0),
            "trigram_surp_cross": (lambda s, k: c.fwd_trigram(s[k - 2], s[k - 1], s[k]), 2, 0),
            "log_bigram_freq_cross": (lambda s, k: c.log_bigram_freq(s[k - 1], s[k]), 1, 0),
        })


@register
class PhonemeTPWithin(_Base):
    name = "phoneme_tp_within"
    columns = ("bigram_surp_within", "trigram_surp_within", "log_bigram_freq_within")
    description = ("Forward phoneme transitional probability with context reset at each "
                   "word boundary (word-initial phone NaN for bigram, first two for trigram).")

    def compute(self, ctx):
        c = _counts(ctx)
        return _score(ctx, within=True, fns={
            "bigram_surp_within": (lambda s, k: c.fwd_bigram(s[k - 1], s[k]), 1, 0),
            "trigram_surp_within": (lambda s, k: c.fwd_trigram(s[k - 2], s[k - 1], s[k]), 2, 0),
            "log_bigram_freq_within": (lambda s, k: c.log_bigram_freq(s[k - 1], s[k]), 1, 0),
        })


@register
class PhonemeTPBackward(_Base):
    name = "phoneme_tp_backward"
    columns = ("bigram_surp_back", "trigram_surp_back")
    description = ("Backward phoneme surprisal across word boundaries: for A B C, bigram "
                   "-ln P(B|C) assigned to B; trigram -ln P(X|Y,Z) assigned to X. NaN for "
                   "the last one / two phones of a sequence.")

    def compute(self, ctx):
        c = _counts(ctx)
        return _score(ctx, within=False, fns={
            "bigram_surp_back": (lambda s, k: c.back_bigram(s[k], s[k + 1]), 0, 1),
            "trigram_surp_back": (lambda s, k: c.back_trigram(s[k], s[k + 1], s[k + 2]), 0, 2),
        })
