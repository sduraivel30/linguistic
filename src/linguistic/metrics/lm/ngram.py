"""5-gram word surprisal (word level, sentences only, context-dependent).

Model: the lm-zoo `ngram` image (SRILM, Kneser-Ney 5-gram trained on English Wikipedia,
closed vocabulary of 43,507 words; https://github.com/cpllab/lm-zoo), run with Apptainer.
The image's own `tokenize` (lowercases, splits contractions: didn't -> did n't) and
`get_surprisals` (SRILM -ppl -debug 2 -order 5) are used. get_surprisals reports bits;
values here are converted to nats. A word split into several tokens gets the mean over
its tokens; the end-of-sentence token is dropped. Words outside the vocabulary are scored
as <unk> and flagged (ngram_oov_5gram = 1). Nonwords are not scored (nearly all <unk>).

Needs `apptainer` (or `singularity`) on PATH and the image in $LINGUISTIC_RESOURCES/lmzoo-ngram/.
"""
from __future__ import annotations

import math
import shutil
import subprocess
import tempfile
from pathlib import Path

import pandas as pd

from ... import resources
from ...registry import Metric, register

LN2 = math.log(2)


def _runtime() -> str:
    exe = shutil.which("apptainer") or shutil.which("singularity")
    if not exe:
        raise RuntimeError("ngram_5gram needs apptainer (or singularity) on PATH, "
                           "e.g. `module load apptainer` on a cluster")
    return exe


def run_container(sif: Path, lines: list[str]) -> tuple[list[list[str]], pd.DataFrame]:
    exe = _runtime()
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / "input.txt"
        f.write_text("\n".join(lines) + "\n")
        base = [exe, "exec", "--contain", "--bind", f"{d}:/data", str(sif)]   # private /tmp
        tok = subprocess.run(base + ["tokenize", "/data/input.txt"], check=True,
                             capture_output=True, text=True).stdout
        sur = subprocess.run(base + ["get_surprisals", "/data/input.txt"], check=True,
                             capture_output=True, text=True).stdout
    tokens = [ln.split() for ln in tok.strip("\n").split("\n")]
    rows = [ln.split("\t") for ln in sur.strip("\n").split("\n")[1:]]
    df = pd.DataFrame(rows, columns=["sentence_id", "token_id", "token", "surprisal_bits"])
    df[["sentence_id", "token_id"]] = df[["sentence_id", "token_id"]].astype(int)
    df["surprisal_bits"] = df["surprisal_bits"].astype(float)
    return tokens, df


def align(words: list[str], tokens: list[str]) -> list[list[int]]:
    """Token indices of each word: consecutive tokens that spell the lowercased word."""
    out, k = [], 0
    for w in words:
        target, got, idx = w.lower(), "", []
        while k < len(tokens) and len(got) < len(target):
            got += tokens[k]
            idx.append(k)
            k += 1
        if got != target:
            raise AssertionError(f"cannot align tokens {tokens} to word {w!r}")
        out.append(idx)
    return out


@register
class NgramSurprisal(Metric):
    name = "ngram_5gram"
    family = "lm"
    level = "word"
    conditions = frozenset({"sentence"})
    nan_reason = "the 5-gram vocabulary has no nonwords"
    context = {"word": "dependent"}
    resources = ("lmzoo-ngram",)
    columns = ("ngram_surprisal_5gram", "ngram_n_tokens_5gram", "ngram_oov_5gram")
    description = ("SRILM Kneser-Ney 5-gram (English Wikipedia, lm-zoo): surprisal of the word "
                   "given the preceding words of its sentence, nats (mean over its tokens); "
                   "number of tokens; 1 if out of vocabulary (<unk>).")

    def compute(self, ctx):
        sif = resources.path("lmzoo-ngram")
        w = ctx.words[ctx.words["condition"] == "sentence"].sort_values(["stimulus_id", "word_idx"])
        groups = list(w.groupby("stimulus_id", sort=False))
        if not groups:
            return pd.DataFrame(columns=["stimulus_id", "word_idx", *self.columns])
        tokens, sur = run_container(sif, [" ".join(g["word"]) for _, g in groups])
        rows = []
        for i, (sid, g) in enumerate(groups):
            s = sur[sur["sentence_id"] == i + 1].sort_values("token_id")
            if len(s) != len(tokens[i]):
                raise AssertionError(f"{sid}: {len(tokens[i])} tokens but {len(s)} surprisals")
            words = g["word"].tolist()
            for wi, idx in zip(g["word_idx"], align(words, [t for t in tokens[i] if t != "</s>"])):
                v = s.iloc[idx]
                rows.append({"stimulus_id": sid, "word_idx": wi,
                             "ngram_surprisal_5gram": float(v["surprisal_bits"].mean()) * LN2,
                             "ngram_n_tokens_5gram": len(idx),
                             "ngram_oov_5gram": float((v["token"] == "<unk>").any())})
        return pd.DataFrame(rows)
