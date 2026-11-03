"""Lexical neighbourhood from edit distance: OLD20 / PLD20, neighbourhood entropy and
surprisal (word level).

Lexicon: every CMUdict word, f(w) = SUBTLEX-US count + 1 (split evenly over a word's
pronunciations for the phonological version), as in the phoneme cohort. D(x, w) is the
Levenshtein distance over letters (orthographic) or phones (phonological) and
d = D / max(len x, len w) its length-normalised form. The word itself (D = 0) is left
out, so real words and nonwords are scored the same way.

    P(w | x)  = f(w) e^(-beta d) / sum_v f(v) e^(-beta d(x, v))     (after Bailey & Hahn 2001)
    entropy   = -sum_w P(w | x) ln P(w | x)
    surprisal = -ln sum_w P_lex(w) e^(-beta d(x, w)),  P_lex(w) = f(w) / sum_v f(v)
    *ld20     = mean raw D to the 20 nearest words (Yarkoni et al. 2008)

Normalising matters: with raw D, the most frequent short words ('a', 'to', 'it') are a
few edits from any short string and dominate the sum, so 'xqzt' would score as more
word-like than 'blick'.
"""
from __future__ import annotations

import numpy as np

BETAS = (10, 20)
K = 20
CHUNK = 256


def neighbourhood(queries: list[str], entries: list[str], freqs: np.ndarray) -> dict[str, np.ndarray]:
    """queries/entries: sequences as strings (one character per letter or phone)."""
    from rapidfuzz.distance import Levenshtein
    from rapidfuzz.process import cdist

    out = {"ld20": [], **{f"entropy_b{b}": [] for b in BETAS}, **{f"surprisal_b{b}": [] for b in BETAS}}
    logf = np.log(freqs)
    log_total = np.log(freqs.sum())
    elen = np.array([len(e) for e in entries], dtype=float)
    for i in range(0, len(queries), CHUNK):
        chunk = queries[i:i + CHUNK]
        raw = cdist(chunk, entries, scorer=Levenshtein.distance, dtype=np.int32, workers=-1).astype(float)
        raw[raw == 0] = np.inf                               # leave the word itself out
        out["ld20"].append(np.sort(raw, axis=1)[:, :K].mean(axis=1))
        d = raw / np.maximum(np.array([len(x) for x in chunk], dtype=float)[:, None], elen[None, :])
        for b in BETAS:
            a = logf - b * d                                  # ln f(w) - beta d
            m = a.max(axis=1, keepdims=True)
            ln_z = m[:, 0] + np.log(np.exp(a - m).sum(axis=1))
            p = np.exp(a - ln_z[:, None])
            out[f"entropy_b{b}"].append(-(p * np.where(p > 0, a - ln_z[:, None], 0)).sum(axis=1))
            out[f"surprisal_b{b}"].append(log_total - ln_z)
    return {k: np.concatenate(v) if v else np.array([]) for k, v in out.items()}


def columns(prefix: str, ld: str) -> tuple[str, ...]:
    return ((f"{prefix}_{ld}20",) + tuple(f"{prefix}_nbhd_entropy_b{b}" for b in BETAS)
            + tuple(f"{prefix}_nbhd_surprisal_b{b}" for b in BETAS))


def rename(values: dict, prefix: str, ld: str) -> dict:
    return {(f"{prefix}_{ld}20" if k == "ld20" else f"{prefix}_nbhd_{k}"): v for k, v in values.items()}


# ---------------------------------------------------------------- orthographic metric
from functools import lru_cache  # noqa: E402


from ...registry import Metric, register  # noqa: E402
from ..lexical.frequency import _subtlex  # noqa: E402
from .ngram import ALPHABET, _NONLETTER  # noqa: E402


@lru_cache(maxsize=2)
def _spelling_lexicon(lexicon):
    lexicon._load()
    counts = _subtlex()
    words = sorted(w for w in lexicon._d if w and set(w) <= ALPHABET)
    return words, np.array([counts.get(w, (0, 0.0))[0] + 1 for w in words], dtype=float)


@register
class OrthographicNeighbourhood(Metric):
    name = "orthographic_neighbourhood"
    family = "orthographic"
    level = "word"
    context = {"word": "dependent"}            # compares the word's letter sequence with the lexicon
    resources = ("subtlex_us",)
    columns = columns("orth", "old")
    description = ("Letter edit distance to CMUdict words (SUBTLEX-US count + 1, word itself left out): "
                   "OLD20; entropy and surprisal (nats) of P(w|x) ∝ f(w) e^(-beta d), d = distance / longer length, beta = 10, 20.")

    def compute(self, ctx):
        words, f = _spelling_lexicon(ctx.lexicon)
        w = ctx.words[["stimulus_id", "word_idx", "word"]].copy()
        q = [_NONLETTER.sub("", x.lower()) for x in w["word"]]
        ok = [bool(x) for x in q]
        vals = rename(neighbourhood([x for x, k in zip(q, ok) if k], words, f), "orth", "old")
        for c in self.columns:
            w[c] = np.nan
            w.loc[ok, c] = vals[c]
        return w[["stimulus_id", "word_idx", *self.columns]]
