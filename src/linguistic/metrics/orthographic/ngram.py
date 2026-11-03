"""Letter n-gram orthographic ill-formedness (word level).

Counts: every CMUdict spelling (word types, lowercased, letters a-z only; entries with
other characters are dropped), padded with '#' at both edges. For a word w1..wn the
letters scored are w1..wn and the end marker, each given its preceding letters (padded
with '#'), with add-one smoothing over the 27 outcomes (26 letters + end):

    P(c | h) = (count(h, c) + 1) / (count(h) + 27)

Column value = mean over those n + 1 positions of -ln P (nats per letter). Higher means
less English-like spelling. Letters outside a-z (apostrophes, hyphens) are dropped
before scoring.
"""
from __future__ import annotations

import math
import re
import string
from collections import Counter
from functools import lru_cache

import pandas as pd

from ...registry import Metric, register

ALPHABET = set(string.ascii_lowercase)
V = len(ALPHABET) + 1          # 26 letters + end marker
_NONLETTER = re.compile(r"[^a-z]")


class LetterCounts:
    def __init__(self, spellings, orders=(2, 3)):
        self.orders = orders
        self.ctx = {n: Counter() for n in orders}
        self.ngram = {n: Counter() for n in orders}
        for w in spellings:
            for n in orders:
                padded = "#" * (n - 1) + w + "#"
                for i in range(n - 1, len(padded)):
                    h = padded[i - n + 1:i]
                    self.ctx[n][h] += 1
                    self.ngram[n][h + padded[i]] += 1

    def neglog(self, n: int, history: str, c: str) -> float:
        return -math.log((self.ngram[n][history + c] + 1) / (self.ctx[n][history] + V))

    def word(self, n: int, word: str) -> float:
        w = _NONLETTER.sub("", word.lower())
        if not w:
            return float("nan")
        padded = "#" * (n - 1) + w + "#"
        vals = [self.neglog(n, padded[i - n + 1:i], padded[i]) for i in range(n - 1, len(padded))]
        return sum(vals) / len(vals)


@lru_cache(maxsize=1)
def _counts(lexicon) -> LetterCounts:
    lexicon._load()
    return LetterCounts(sorted(w for w in lexicon._d if w and set(w) <= ALPHABET))


@register
class OrthographicNgram(Metric):
    name = "orthographic_ngram"
    family = "orthographic"
    level = "word"
    context = {"word": "dependent"}            # scores the word's letter sequence
    columns = ("orth_bigram_surp", "orth_trigram_surp")
    description = ("Spelling ill-formedness: mean over the word's letters and its end of "
                   "-ln P(letter | previous 1 (bigram) or 2 (trigram) letters), counts from "
                   "CMUdict spellings, add-one smoothing; nats per letter, higher = less English-like.")

    def compute(self, ctx):
        c = _counts(ctx.lexicon)
        w = ctx.words[["stimulus_id", "word_idx", "word"]].copy()
        w["orth_bigram_surp"] = [c.word(2, x) for x in w["word"]]
        w["orth_trigram_surp"] = [c.word(3, x) for x in w["word"]]
        return w[["stimulus_id", "word_idx", *self.columns]]
