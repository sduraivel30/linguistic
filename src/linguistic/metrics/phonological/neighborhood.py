"""Phonological neighbourhood: PLD20, neighbourhood entropy and surprisal over phone
edit distance. Same definitions as orthographic/neighborhood.py, with each CMUdict
pronunciation an entry (f split evenly over a word's pronunciations)."""
from __future__ import annotations

from functools import lru_cache

import numpy as np

from ...registry import Metric, register
from ..lexical.frequency import _subtlex
from ..orthographic.neighborhood import columns, neighbourhood, rename


class _Codes(dict):
    """phone -> one private-use character, assigned on first sight."""
    def __missing__(self, p):
        self[p] = chr(0xE000 + len(self))
        return self[p]


@lru_cache(maxsize=2)
def _pron_lexicon(lexicon):
    lexicon._load()
    counts = _subtlex()
    codes, entries, f = _Codes(), [], []
    for word, prons in sorted(lexicon._d.items()):
        c = (counts.get(word.lower(), (0, 0.0))[0] + 1) / len(prons)
        for ph in prons:
            entries.append("".join(codes[p] for p in ph))
            f.append(c)
    return codes, entries, np.array(f)


@register
class PhonologicalNeighbourhood(Metric):
    name = "phonological_neighbourhood"
    family = "phonological"
    level = "word"
    context = {"word": "dependent"}            # compares the word's phone sequence with the lexicon
    requires = frozenset({"phones"})
    resources = ("subtlex_us",)
    columns = columns("phone", "pld")
    description = ("Phone edit distance to CMUdict pronunciations (SUBTLEX-US count + 1, the word's own "
                   "pronunciation left out): PLD20; entropy and surprisal (nats) of P(w|x) ∝ f(w) e^(-beta d), "
                   "d = distance / longer length, beta = 10, 20.")

    def compute(self, ctx):
        codes, entries, f = _pron_lexicon(ctx.lexicon)
        w = ctx.words[["stimulus_id", "word_idx", "phones"]].copy()
        w = w[w["phones"].fillna("") != ""]
        q = ["".join(codes[p] for p in s.split()) for s in w["phones"]]
        vals = rename(neighbourhood(q, entries, f), "phone", "pld")
        for c in self.columns:
            w[c] = vals[c]
        return w[["stimulus_id", "word_idx", *self.columns]]
