"""Scoring for masked (encoder-only), XLNet and T5 models.

masked  Whole-word pseudo-log-likelihood, left to right within the word (Kauf & Ivanova
        2023, ACL; Salazar et al. 2020): to score sub-token j of a word, mask it and the
        word's later sub-tokens; every other word stays visible on both sides.
        Entropy: of the prediction for the first sub-token with the whole word masked.
xlnet   Left to right: sub-token t is predicted with a permutation mask that hides t and
        everything after it. XLNet predicts poorly from a short context, so (as in the
        Hugging Face XLNet examples) a fixed padding passage ending in <eod> comes before the
        stimulus. Entropy: at the word's first sub-token.
t5      The word is replaced by the sentinel <extra_id_0> and the decoder produces
        "<extra_id_0> word <extra_id_1>"; surprisal over the word's tokens. Both sides
        visible. Entropy: of the decoder's prediction for the word's first token.
Every function returns one row per word: surprisal (mean over sub-tokens, nats), entropy
(nats), n_subtokens.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _logp(logits):
    import torch.nn.functional as F
    return F.log_softmax(logits.float(), dim=-1)


def _entropy(lp) -> float:
    return float(-(lp.exp() * lp).sum())


def word_spans(enc, n_words: int) -> list[list[int]]:
    """Token positions of each word from a fast tokenizer's word_ids()."""
    spans = [[] for _ in range(n_words)]
    for pos, w in enumerate(enc.word_ids()):
        if w is not None:
            spans[w].append(pos)
    if any(not s for s in spans):
        raise ValueError("a word encodes to no tokens")
    return spans


def score_masked(words: list[str], tok, model) -> pd.DataFrame:
    import torch
    enc = tok(words, is_split_into_words=True)
    ids = torch.tensor(enc["input_ids"], device=model.device)
    spans = word_spans(enc, len(words))
    batch, targets = [], []
    for wi, span in enumerate(spans):
        for j, p in enumerate(span):
            x = ids.clone()
            x[span[j:]] = tok.mask_token_id
            batch.append(x)
            targets.append((wi, j, p))
    with torch.inference_mode():
        logits = model(input_ids=torch.stack(batch)).logits
    nll = [[] for _ in words]
    ent = [np.nan] * len(words)
    for b, (wi, j, p) in enumerate(targets):
        lp = _logp(logits[b, p])
        nll[wi].append(-float(lp[ids[p]]))
        if j == 0:
            ent[wi] = _entropy(lp)
    return pd.DataFrame({"surprisal": [float(np.mean(v)) for v in nll], "entropy": ent,
                         "n_subtokens": [len(s) for s in spans]})


XLNET_PADDING = (
    "In 1991, the remains of Russian Tsar Nicholas II and his family (except for Alexei and "
    "Maria) are discovered. The voice of Nicholas's young son, Tsarevich Alexei Nikolaevich, "
    "narrates the remainder of the story. 1883 Western Siberia, a young Grigori Rasputin is "
    "asked by his father and a group of men to perform magic. Rasputin has a vision and "
    "denounces one of the men as a horse thief. Although his father initially slaps him for "
    "making such an accusation, Rasputin watches as the man is chased outside and beaten. "
    "Twenty years later, Rasputin sees a vision of the Virgin Mary, prompting him to become a "
    "priest. Rasputin quickly becomes famous, with people, even a bishop, begging for his "
    "blessing. <eod>")


def xlnet_prefix(tok) -> list[int]:
    return tok(XLNET_PADDING, add_special_tokens=False)["input_ids"]


def score_xlnet(words: list[str], tok, model) -> pd.DataFrame:
    import torch
    enc = tok(words, is_split_into_words=True, add_special_tokens=False)
    prefix = xlnet_prefix(tok)
    spans = [[p + len(prefix) for p in s] for s in word_spans(enc, len(words))]
    ids = torch.tensor(prefix + enc["input_ids"], device=model.device)
    L = len(ids)
    pos = [p for s in spans for p in s]
    perm = torch.zeros(len(pos), L, L, device=model.device)
    tmap = torch.zeros(len(pos), 1, L, device=model.device)
    for b, p in enumerate(pos):
        perm[b, :, p:] = 1.0                       # nobody sees p or anything after it
        tmap[b, 0, p] = 1.0
    with torch.inference_mode():
        logits = model(input_ids=ids.expand(len(pos), L), perm_mask=perm, target_mapping=tmap).logits[:, 0]
    by_pos = {p: _logp(logits[b]) for b, p in enumerate(pos)}
    return pd.DataFrame({
        "surprisal": [float(np.mean([-float(by_pos[p][ids[p]]) for p in s])) for s in spans],
        "entropy": [_entropy(by_pos[s[0]]) for s in spans],
        "n_subtokens": [len(s) for s in spans]})


def score_t5(words: list[str], tok, model) -> pd.DataFrame:
    import torch
    s0, s1 = "<extra_id_0>", "<extra_id_1>"
    rows = []
    for i, w in enumerate(words):
        src = " ".join(words[:i] + [s0] + words[i + 1:])
        tgt = tok(f"{s0} {w} {s1}").input_ids              # [s0, word..., s1, </s>]
        if tgt[0] != tok.convert_tokens_to_ids(s0):
            raise AssertionError(f"unexpected T5 target encoding for {w!r}")
        k = tgt.index(tok.convert_tokens_to_ids(s1))
        x = torch.tensor([tok(src).input_ids], device=model.device)
        y = torch.tensor([tgt], device=model.device)
        with torch.inference_mode():
            logits = model(input_ids=x, labels=y).logits[0]  # row t predicts label t
        lp = _logp(logits)
        rows.append({"surprisal": float(np.mean([-float(lp[t, tgt[t]]) for t in range(1, k)])),
                     "entropy": _entropy(lp[1]), "n_subtokens": k - 1})
    return pd.DataFrame(rows)
