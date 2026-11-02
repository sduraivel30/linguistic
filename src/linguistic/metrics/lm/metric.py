"""One word-level metric per model in models.MODELS: lm_<name>."""
from __future__ import annotations

import pandas as pd

from ...registry import Metric, register
from . import bidirectional, causal
from .models import MODELS, ModelSpec, load

SCORERS = {"causal": causal.score, "masked": bidirectional.score_masked,
           "xlnet": bidirectional.score_xlnet, "t5": bidirectional.score_t5}
SEES = {"causal": "preceding words", "xlnet": "preceding words",
        "masked": "words on both sides", "t5": "words on both sides"}


def make_metric(spec: ModelSpec) -> type[Metric]:
    class _LM(Metric):
        family = "lm"
        level = "word"
        context = {"word": "dependent"}
        description = (f"{spec.repo}@{spec.revision[:7]} ({spec.kind}; the word is scored "
                       f"given the {SEES[spec.kind]} of its stimulus): mean sub-token surprisal "
                       "and entropy of the prediction for the word (nats).")

        def compute(self, ctx):
            tok, model, _ = load(spec.name)
            frames = []
            for sid, g in ctx.words.sort_values(["stimulus_id", "word_idx"]).groupby("stimulus_id", sort=False):
                v = SCORERS[spec.kind](g["word"].tolist(), tok, model)
                v.columns = list(self.columns)
                v.insert(0, "word_idx", g["word_idx"].to_numpy())
                v.insert(0, "stimulus_id", sid)
                frames.append(v)
            return pd.concat(frames, ignore_index=True)

    _LM.name = f"lm_{spec.name}"
    _LM.columns = (f"lm_surprisal_{spec.tag}", f"lm_entropy_{spec.tag}", f"lm_n_subtokens_{spec.tag}")
    _LM.resources = (spec.name,)
    _LM.__name__ = f"LM_{spec.tag}"
    return _LM


for _spec in MODELS.values():
    register(make_metric(_spec))
