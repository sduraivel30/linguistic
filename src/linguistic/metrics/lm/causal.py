"""Decoder-only (causal) scoring: each word given the preceding words.

Vendored from Speech_Lang_iEEG @ a79857d, surprisal/linguistic/code/
(linguistic_surprisal_utils.tokenize_aligned_words, get_bos_token_id;
extract_linguistic_surprisal.score_input_ids):
  - context resets at every stimulus; an explicit BOS (else EOS) token precedes word 1
  - word i>1 is encoded with a leading space; the piecewise encoding must equal the
    whole-string encoding (asserted)
  - token surprisal = -log p(token | preceding tokens), natural log, float32;
    the mean token NLL must match the model's own loss within 2e-3 (asserted)
New here: entropy before the word.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

LOSS_TOL = 2e-3


def bos_id(tok) -> int:
    i = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
    if i is None:
        raise ValueError("tokenizer has neither a BOS nor an EOS token")
    return int(i)


def tokenize(words: list[str], tok) -> tuple[list[int], list[tuple[int, int]]]:
    ids, spans = [bos_id(tok)], []
    for i, w in enumerate(words):
        piece = tok.encode(("" if i == 0 else " ") + w, add_special_tokens=False)
        if not piece:
            raise ValueError(f"word {w!r} encodes to no tokens")
        spans.append((len(ids), len(ids) + len(piece)))
        ids += piece
    whole = tok.encode(" ".join(words), add_special_tokens=False)
    if ids[1:] != whole:
        raise AssertionError("piecewise tokenization differs from whole-string tokenization "
                             f"for {' '.join(words)!r}")
    return ids, spans


def score(words: list[str], tok, model) -> pd.DataFrame:
    """Per word: mean sub-token surprisal, entropy before the word, n sub-tokens."""
    import torch
    import torch.nn.functional as F
    ids, spans = tokenize(words, tok)
    n_ctx = getattr(model.config, "n_positions", None) or getattr(model.config, "max_position_embeddings", None)
    if n_ctx and len(ids) > n_ctx:
        raise ValueError(f"stimulus is {len(ids)} tokens, longer than the model's {n_ctx}")
    x = torch.tensor([ids], dtype=torch.long, device=model.device)
    with torch.inference_mode():
        out = model(input_ids=x, labels=x, use_cache=False, return_dict=True)
        logp = F.log_softmax(out.logits[0, :-1, :].float(), dim=-1)   # row t predicts token t+1
        nll = -logp.gather(1, x[0, 1:, None])[:, 0]
        ent = -(logp.exp() * logp).sum(-1)
    nll = nll.cpu().numpy().astype(np.float64)
    ent = ent.cpu().numpy().astype(np.float64)
    err = abs(float(nll.mean()) - float(out.loss))
    if err > LOSS_TOL:
        raise AssertionError(f"token NLL mean differs from model loss by {err:.3g}")
    return pd.DataFrame([{"surprisal": float(nll[s - 1:e - 1].mean()), "entropy": float(ent[s - 1]),
                          "n_subtokens": e - s} for s, e in spans])
