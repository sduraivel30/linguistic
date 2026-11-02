"""The language models this package knows, how to fetch them, and how to load them.

kind decides how a model is scored and read:
  causal  decoder-only, scored left to right                  (AutoModelForCausalLM)
  masked  encoder-only masked LM, both sides of the word        (AutoModelForMaskedLM)
  xlnet   permutation LM, scored left to right with a perm mask (XLNetLMHeadModel)
  t5      encoder-decoder, the word filled into a sentinel      (T5ForConditionalGeneration)
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache

from ... import resources

README = "src/linguistic/metrics/lm/README.md"


@dataclass(frozen=True)
class ModelSpec:
    name: str
    kind: str
    repo: str
    revision: str
    tag: str              # column suffix
    weights: str = "model.safetensors"


MODELS = {m.name: m for m in [
    ModelSpec("gpt2-xl", "causal", "openai-community/gpt2-xl",
              "15ea56dee5df4983c59b2538573817e1667135e2", "gpt2xl"),
    ModelSpec("albert-xxlarge-v1", "masked", "albert/albert-xxlarge-v1",
              "43129068ee5f6a481c148daeac11cc593b8ff440", "albertxxl", weights="pytorch_model.bin"),
    ModelSpec("bert-large-uncased", "masked", "google-bert/bert-large-uncased",
              "6da4b6a26a1877e173fca3225479512db81a5e5b", "bertlarge"),
    ModelSpec("xlnet-large-cased", "xlnet", "xlnet/xlnet-large-cased",
              "37658b4f179eaf971d127d16bfbd9ca676a93034", "xlnetlarge", weights="pytorch_model.bin"),
    ModelSpec("t5-large", "t5", "google-t5/t5-large",
              "150ebc2c4b72291e770f58e6057481c8d2ed331a", "t5large"),
]}

for _m in MODELS.values():
    resources.register_model(_m.name, _m.repo, _m.revision, README,
                             patterns=("*.json", "*.txt", "*.model", _m.weights))


def add_model(spec: ModelSpec) -> None:
    """Register a model at run time (also used by the tests)."""
    MODELS[spec.name] = spec
    resources.register_model(spec.name, spec.repo, spec.revision, README,
                             patterns=("*.json", "*.txt", "*.model", spec.weights))


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


_CLASSES = {"causal": "AutoModelForCausalLM", "masked": "AutoModelForMaskedLM",
            "xlnet": "XLNetLMHeadModel", "t5": "T5ForConditionalGeneration"}


@lru_cache(maxsize=2)
def load(name: str):
    """(tokenizer, model, spec) from $LINGUISTIC_RESOURCES/<name>/, never downloading."""
    import torch
    import transformers
    spec = MODELS[name]
    d = str(resources.path(name).parent)
    tok = transformers.AutoTokenizer.from_pretrained(d, local_files_only=True, use_fast=True)
    cls = getattr(transformers, _CLASSES[spec.kind])
    model = cls.from_pretrained(d, local_files_only=True, dtype=torch.float32).to(device())
    model.eval()
    return tok, model, spec
