"""Contextual language-model metrics from causal (left-to-right) Hugging Face LMs.

Scoring vendored from Speech_Lang_iEEG @ a79857d, surprisal/linguistic/code/
(linguistic_surprisal_utils.tokenize_aligned_words, get_bos_token_id;
extract_linguistic_surprisal.score_input_ids):
  - context resets at every stimulus; an explicit BOS (else EOS) token precedes word 1
  - words are the stimulus's tokens joined by one space, case kept, edge punctuation removed;
    word i>1 is encoded with a leading space; the piecewise encoding must equal the
    whole-string encoding (asserted)
  - token surprisal = -log p(token | preceding tokens), natural log, float32 model;
    the mean token NLL must match the model's own loss within 2e-3 (asserted)
Measures (per user decision, summed surprisal is not reported):
  lm_surprisal_<tag>  mean surprisal over the word's sub-tokens (nats)
  lm_entropy_<tag>    entropy (nats) of the model's next-token distribution at the position
                      that predicts the word's first sub-token, i.e. its uncertainty just
                      before the word
  lm_n_subtokens_<tag>
New here: entropy, the model list, device selection.
"""
from __future__ import annotations

import os
from functools import lru_cache

import numpy as np
import pandas as pd

from ... import resources
from ...registry import Metric, register

README = "src/linguistic/metrics/lm/README.md"
LOSS_TOL = 2e-3

# name -> (Hugging Face repo, pinned revision, column tag)
MODELS = {
    "gpt2-xl": ("openai-community/gpt2-xl", "15ea56dee5df4983c59b2538573817e1667135e2", "gpt2xl"),
}


def device():
    import torch
    want = os.environ.get("LINGUISTIC_DEVICE")
    if want:
        return torch.device(want)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@lru_cache(maxsize=2)
def load(model_dir: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(model_dir, local_files_only=True,
                                                 dtype=torch.float32).to(device())
    model.eval()
    return tok, model


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
        logits = out.logits[0, :-1, :].float()                # position t predicts token t+1
        logp = F.log_softmax(logits, dim=-1)
        nll = -logp.gather(1, x[0, 1:, None])[:, 0]
        ent = -(logp.exp() * logp).sum(-1)
    nll = nll.cpu().numpy().astype(np.float64)
    ent = ent.cpu().numpy().astype(np.float64)
    err = abs(float(nll.mean()) - float(out.loss))
    if err > LOSS_TOL:
        raise AssertionError(f"token NLL mean differs from model loss by {err:.3g}")
    rows = []
    for s, e in spans:                    # token at position p is predicted at row p - 1
        rows.append({"surprisal": float(nll[s - 1:e - 1].mean()), "entropy": float(ent[s - 1]),
                     "n_subtokens": e - s})
    return pd.DataFrame(rows)


def _make(name: str, repo: str, revision: str, tag: str) -> None:
    resources.register_model(name, repo, revision, README)

    class _LM(Metric):
        family = "lm"
        level = "word"
        description = (f"{repo}@{revision[:7]}: mean sub-token surprisal of the word given the "
                       "preceding words of its stimulus, and the entropy of the next-token "
                       "distribution just before the word (nats).")

        def compute(self, ctx):
            tok, model = load(str(resources.path(name).parent))
            frames = []
            for sid, g in ctx.words.sort_values(["stimulus_id", "word_idx"]).groupby("stimulus_id", sort=False):
                v = score(g["word"].tolist(), tok, model)
                v.columns = list(self.columns)
                v.insert(0, "word_idx", g["word_idx"].to_numpy())
                v.insert(0, "stimulus_id", sid)
                frames.append(v)
            return pd.concat(frames, ignore_index=True)

    _LM.name = f"lm_{name}"
    _LM.columns = (f"lm_surprisal_{tag}", f"lm_entropy_{tag}", f"lm_n_subtokens_{tag}")
    _LM.resources = (name,)
    _LM.__name__ = f"LM_{tag}"
    register(_LM)


for _n, (_repo, _rev, _tag) in MODELS.items():
    _make(_n, _repo, _rev, _tag)
