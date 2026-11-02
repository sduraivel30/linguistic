"""Static word embeddings (GloVe): context-independent, one vector per word.

GloVe 840B, cased, 300 dimensions (Pennington et al. 2014;
https://nlp.stanford.edu/projects/glove/). `linguistic fetch glove-840b` downloads
glove.840B.300d.zip (2.0 GB) and converts it once into vocab.txt + vectors.npy
(float32, 2.2 M x 300, 2.6 GB) in $LINGUISTIC_RESOURCES/glove-840b/.
Lookup: the word as written, else lowercased; otherwise out of vocabulary (NaN vector).
"""
from __future__ import annotations

import io
import zipfile
from functools import lru_cache
from pathlib import Path

import numpy as np

from . import resources

DIM = 300
STATIC = {"glove-840b"}


def convert_glove(zip_path: Path, out_dir: Path) -> Path:
    """glove.840B.300d.zip -> vocab.txt + vectors.npy (keeps the first vector of a duplicate)."""
    vocab, rows, seen = [], [], set()
    with zipfile.ZipFile(zip_path) as z:
        member = next(m for m in z.namelist() if m.endswith(".txt"))
        with z.open(member) as f:
            for line in io.TextIOWrapper(f, encoding="utf-8"):
                parts = line.rstrip("\n").split(" ")
                word = " ".join(parts[:-DIM])            # a few tokens contain spaces
                if word in seen or len(parts) <= DIM:
                    continue
                seen.add(word)
                vocab.append(word)
                rows.append(np.asarray(parts[-DIM:], dtype=np.float32))
    np.save(out_dir / "vectors.npy", np.stack(rows))
    (out_dir / "vocab.txt").write_text("\n".join(vocab) + "\n", encoding="utf-8")
    return out_dir / "vectors.npy"


@lru_cache(maxsize=1)
def load_glove():
    d = resources.path("glove-840b").parent
    vocab = (d / "vocab.txt").read_text(encoding="utf-8").split("\n")[:-1]
    return {w: i for i, w in enumerate(vocab)}, np.load(d / "vectors.npy", mmap_mode="r")


def lookup(words: list[str]) -> tuple[np.ndarray, list[bool]]:
    index, vectors = load_glove()
    out = np.full((len(words), DIM), np.nan, dtype=np.float32)
    found = []
    for i, w in enumerate(words):
        j = index.get(w, index.get(w.lower()))
        found.append(j is not None)
        if j is not None:
            out[i] = vectors[j]
    return out, found
