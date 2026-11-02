import math

import pandas as pd
import pytest

from linguistic.assemble import run_metrics
from linguistic.io import load_stimuli
from linguistic.normalize import build_units
from linguistic.registry import Context, Metric, register, select, unregister

from conftest import write_stimuli


class _Letters(Metric):
    name, family, level = "t_letters", "t_family", "word"
    conditions = frozenset({"sentence"})
    context = {"word": "independent"}

    def compute(self, ctx):
        return ctx.words.assign(t_letters=ctx.words["word"].str.len())[["stimulus_id", "word_idx", "t_letters"]]


class _PhoneIndex(Metric):
    name, family, level = "t_phone_idx", "t_family", "phone"
    requires = frozenset({"phones"})
    context = {"phone": "dependent", "word": "independent"}

    def compute(self, ctx):
        p = ctx.phones
        return p.assign(t_phone_idx=p["phone_in_word"].astype(float))[
            ["stimulus_id", "word_idx", "phone_in_word", "t_phone_idx"]]


@pytest.fixture
def metrics():
    register(_Letters)
    register(_PhoneIndex)
    yield select(["t_family"])
    unregister("t_letters")
    unregister("t_phone_idx")


@pytest.fixture
def ctx(tmp_path, lexicon):
    s = load_stimuli(write_stimuli(tmp_path / "s.csv", [
        {"stimulus_id": "s1", "condition": "sentence", "text": "a cat sat"},
        {"stimulus_id": "n1", "condition": "nonword", "text": "zorpleth cat"},
    ]))
    u = build_units(s, lexicon)
    return Context(stimuli=s, words=u.words, phones=u.phones, lexicon=lexicon)


def test_condition_mask_and_stimulus_mean(ctx, metrics):
    t = run_metrics(ctx, metrics)
    w = t.words.set_index(["stimulus_id", "word_idx"])
    assert w.loc[("s1", 2), "t_letters"] == 3
    assert w.loc[["n1"], "t_letters"].isna().all()                    # sentence-only metric
    st = t.stimuli.set_index("stimulus_id")
    assert math.isclose(st.loc["s1", "t_letters"], (1 + 3 + 3) / 3)
    assert st.loc["s1", "n_valid_t_letters"] == 3
    assert st.loc["n1", "n_valid_t_letters"] == 0


def test_phone_metric_is_mean_of_word_means(ctx, metrics):
    t = run_metrics(ctx, metrics)
    w = t.words.set_index(["stimulus_id", "word_idx"])
    # "a" = AH (1 phone) -> 1; "cat" = K AE T -> mean(1,2,3) = 2; "sat" -> 2
    assert w.loc[("s1", 1), "t_phone_idx"] == 1
    assert w.loc[("s1", 2), "t_phone_idx"] == 2
    st = t.stimuli.set_index("stimulus_id")
    assert math.isclose(st.loc["s1", "t_phone_idx"], (1 + 2 + 2) / 3)   # not the all-phone mean 1.857
    # "zorpleth" has no phones -> NaN, excluded from n_valid
    assert math.isnan(w.loc[("n1", 1), "t_phone_idx"])
    assert st.loc["n1", "n_valid_t_phone_idx"] == 1


def test_duplicate_registration_and_unknown_names(metrics):
    with pytest.raises(TypeError, match="twice"):
        register(_Letters)
    with pytest.raises(KeyError, match="unknown metric"):
        select(["nope"])


def test_column_clash(ctx, metrics):
    class _Clash(Metric):
        name, family, level, columns = "t_clash", "t_other", "word", ("word",)

        def compute(self, ctx):
            return pd.DataFrame()

    with pytest.raises(ValueError, match="already exist"):
        run_metrics(ctx, [_Clash()])


def test_level_context_selector(metrics):
    assert [m.name for m in select(["phone:dependent"]) if m.family == "t_family"] == ["t_phone_idx"]
    assert {m.name for m in select(["word:independent"]) if m.family == "t_family"} == {"t_letters", "t_phone_idx"}
