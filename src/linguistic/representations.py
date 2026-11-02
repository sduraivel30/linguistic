"""Layer-wise word activations and static word embeddings from the language models.

For every word of every stimulus, and every layer (the embedding layer + each block):
  activations.npy   (n_words, n_layers + 1, hidden)  contextual hidden state of the word
  embeddings.npy    (n_words, hidden)                the model's input (token) embedding of
                                                     the word: no context, no position
  stimuli_last.npy  (n_stimuli, n_layers + 1, hidden) activation of the stimulus's last word
  stimuli_mean.npy  (n_stimuli, n_layers + 1, hidden) mean over the stimulus's words
  words.csv, stimuli.csv, layers.csv                  row indices for the arrays
  manifest.json

context: 'incremental' (default) -- word i is read with words 1..i only, so no model sees
         later words (as in a word-by-word presentation); 'full' -- the whole stimulus at
         once (for bidirectional models later words are then visible). Causal models give
         the same values either way.
pool:    'last' (default) -- the word's last sub-token, 'mean' -- mean over its sub-tokens.
T5: the encoder's hidden states. XLNet: read after the same padding passage as its scoring.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import __version__
from .manifest import _git_commit, sha256
from .metrics.lm import causal
from .metrics.lm.bidirectional import word_spans, xlnet_prefix
from .metrics.lm.models import MODELS, load


def encode(words: list[str], tok, kind: str):
    """(input_ids list, token positions of each word)."""
    if kind == "causal":
        ids, spans = causal.tokenize(words, tok)
        return ids, [list(range(s, e)) for s, e in spans]
    if kind == "xlnet":
        enc = tok(words, is_split_into_words=True, add_special_tokens=False)
        prefix = xlnet_prefix(tok)                          # same padding passage as scoring
        return prefix + enc["input_ids"], [[p + len(prefix) for p in s] for s in word_spans(enc, len(words))]
    enc = tok(words, is_split_into_words=True)              # masked: [CLS] .. [SEP]; t5: .. </s>
    return enc["input_ids"], word_spans(enc, len(words))


def hidden_states(ids: list[int], model, kind: str):
    import torch
    x = torch.tensor([ids], device=model.device)
    with torch.inference_mode():
        if kind == "t5":
            hs = model.encoder(input_ids=x, output_hidden_states=True).hidden_states
        else:
            hs = model(input_ids=x, output_hidden_states=True).hidden_states
    return torch.stack([h[0] for h in hs]).float()          # (n_layers + 1, tokens, hidden)


def word_vectors(words: list[str], tok, model, kind: str, context: str, pool: str):
    """(activations (n_words, L+1, d), embeddings (n_words, d)) for one stimulus."""
    import torch
    reduce = (lambda h, s: h[:, s[-1]]) if pool == "last" else (lambda h, s: h[:, s].mean(1))
    ids, spans = encode(words, tok, kind)
    acts = []
    if context == "full" or kind == "causal":
        h = hidden_states(ids, model, kind)
        acts = [reduce(h, s) for s in spans]
    else:
        for i in range(len(words)):
            pids, pspans = encode(words[:i + 1], tok, kind)
            acts.append(reduce(hidden_states(pids, model, kind), pspans[i]))
    emb_layer = model.get_input_embeddings()
    with torch.inference_mode():
        emb = [emb_layer(torch.tensor([ids[p] for p in s], device=model.device)).float().mean(0) for s in spans]
    return torch.stack(acts).cpu().numpy(), torch.stack(emb).cpu().numpy()


def represent_static(stimuli_path, words: pd.DataFrame, stimuli: pd.DataFrame, name: str, out_dir,
                     dtype: str = "float32") -> Path:
    """GloVe: embeddings.npy (n_words, 300), stimuli_last/mean.npy (n_stimuli, 300)."""
    from .static_embeddings import lookup
    out = Path(out_dir) / name
    out.mkdir(parents=True, exist_ok=True)
    w = words.sort_values(["stimulus_id", "word_idx"], kind="stable").reset_index(drop=True)
    vec, found = lookup(w["word"].tolist())
    np.save(out / "embeddings.npy", vec.astype(dtype))
    order = list(dict.fromkeys(w["stimulus_id"]))
    starts = w.groupby("stimulus_id", sort=False).indices
    last = np.stack([vec[starts[s][-1]] for s in order])
    with np.errstate(invalid="ignore"):
        mean = np.stack([np.nanmean(vec[starts[s]], axis=0) if np.isfinite(vec[starts[s]]).any()
                         else np.full(vec.shape[1], np.nan, np.float32) for s in order])
    np.save(out / "stimuli_last.npy", last.astype(dtype))
    np.save(out / "stimuli_mean.npy", mean.astype(dtype))
    w.assign(row=range(len(w)), in_vocab=found)[["row", "stimulus_id", "condition", "word_idx", "word", "in_vocab"]].to_csv(out / "words.csv", index=False)
    st = stimuli.set_index("stimulus_id").loc[order, ["condition"]].reset_index()
    st.insert(0, "row", range(len(st)))
    st.to_csv(out / "stimuli.csv", index=False)
    (out / "manifest.json").write_text(json.dumps({
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "linguistic_version": __version__, "linguistic_git_commit": _git_commit(),
        "input": {"path": str(stimuli_path), "sha256": sha256(stimuli_path)},
        "model": {"name": name, "kind": "static", "source": "https://nlp.stanford.edu/data/glove.840B.300d.zip"},
        "lookup": "as written, else lowercase; out of vocabulary = NaN (stimulus mean skips them)",
        "n_out_of_vocab": int(len(found) - sum(found)), "dtype": dtype,
        "shapes": {"embeddings": list(vec.shape), "stimuli_last": list(last.shape), "stimuli_mean": list(mean.shape)},
    }, indent=2))
    return out


def represent(stimuli_path, words: pd.DataFrame, stimuli: pd.DataFrame, name: str, out_dir,
              context: str = "incremental", pool: str = "last", dtype: str = "float32") -> Path:
    from .static_embeddings import STATIC
    if name in STATIC:
        return represent_static(stimuli_path, words, stimuli, name, out_dir, dtype=dtype)
    if context not in ("incremental", "full") or pool not in ("last", "mean"):
        raise ValueError("context must be incremental|full and pool last|mean")
    tok, model, spec = load(name)
    out = Path(out_dir) / name
    out.mkdir(parents=True, exist_ok=True)
    w = words.sort_values(["stimulus_id", "word_idx"], kind="stable").reset_index(drop=True)
    order = list(dict.fromkeys(w["stimulus_id"]))
    acts = emb = None
    rows = 0
    for sid in order:
        g = w[w["stimulus_id"] == sid]
        a, e = word_vectors(g["word"].tolist(), tok, model, spec.kind, context, pool)
        if acts is None:
            acts = np.lib.format.open_memmap(out / "activations.npy", mode="w+", dtype=dtype,
                                             shape=(len(w),) + a.shape[1:])
            emb = np.lib.format.open_memmap(out / "embeddings.npy", mode="w+", dtype=dtype,
                                            shape=(len(w), e.shape[1]))
        acts[rows:rows + len(a)] = a
        emb[rows:rows + len(e)] = e
        rows += len(a)
    acts.flush()
    emb.flush()

    starts = w.groupby("stimulus_id", sort=False).indices
    last = np.stack([acts[starts[s][-1]] for s in order]).astype(dtype)
    mean = np.stack([np.asarray(acts[starts[s]], dtype=np.float32).mean(0) for s in order]).astype(dtype)
    np.save(out / "stimuli_last.npy", last)
    np.save(out / "stimuli_mean.npy", mean)

    w.assign(row=range(len(w)))[["row", "stimulus_id", "condition", "word_idx", "word"]].to_csv(out / "words.csv", index=False)
    st = stimuli.set_index("stimulus_id").loc[order, ["condition"]].reset_index()
    st.insert(0, "row", range(len(st)))
    st.to_csv(out / "stimuli.csv", index=False)
    n_layers = acts.shape[1]
    pd.DataFrame({"layer": range(n_layers),
                  "name": ["embedding"] + [f"layer_{k}" for k in range(1, n_layers)]}).to_csv(out / "layers.csv", index=False)
    (out / "manifest.json").write_text(json.dumps({
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "linguistic_version": __version__, "linguistic_git_commit": _git_commit(),
        "input": {"path": str(stimuli_path), "sha256": sha256(stimuli_path)},
        "model": {"name": name, "kind": spec.kind, "hf_repo": spec.repo, "hf_revision": spec.revision},
        "context": context, "pool": pool, "dtype": dtype,
        "shapes": {"activations": list(acts.shape), "embeddings": list(emb.shape),
                   "stimuli_last": list(last.shape), "stimuli_mean": list(mean.shape)},
    }, indent=2))
    return out


def available_models() -> list[str]:
    return list(MODELS)
