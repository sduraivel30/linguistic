import math

import pytest

from linguistic.assemble import run_metrics
from linguistic.io import load_stimuli
from linguistic.metrics.phonological.ngram import PhonemeCounts
from linguistic.normalize import build_units
from linguistic.registry import Context, select

from conftest import write_stimuli


@pytest.fixture(scope="module")
def counts(lexicon):
    return PhonemeCounts(lexicon)


def _run(tmp_path, lexicon, text, condition="sentence"):
    s = load_stimuli(write_stimuli(tmp_path / "s.csv", [
        {"stimulus_id": "x", "condition": condition, "text": text}]))
    u = build_units(s, lexicon)
    return run_metrics(Context(s, u.words, u.phones, lexicon), select(["phonological"]))


def test_counts_definitions(counts):
    assert math.isclose(counts.fwd_bigram("K", "AE"),
                        -math.log(counts.bi[("K", "AE")] / counts.uni["K"]))
    assert math.isclose(counts.back_bigram("K", "AE"),
                        -math.log(counts.bi[("K", "AE")] / counts.uni["AE"]))
    assert math.isnan(counts.fwd_bigram("NG", "NG"))           # unseen -> NaN


def test_cross_within_backward_positions(tmp_path, lexicon, counts):
    # "cat sat" = K AE T | S AE T
    p = _run(tmp_path, lexicon, "cat sat").phones
    s = p[p["word_idx"] == 2].iloc[0]                           # S, word-initial
    assert math.isclose(s["bigram_surp_cross"], counts.fwd_bigram("T", "S"))
    assert math.isnan(s["bigram_surp_within"])                  # context reset at the boundary
    assert math.isclose(s["trigram_surp_cross"], counts.fwd_trigram("AE", "T", "S"))
    t1 = p[(p["word_idx"] == 1) & (p["phone_in_word"] == 3)].iloc[0]   # T of "cat"
    assert math.isclose(t1["bigram_surp_back"], counts.back_bigram("T", "S"))
    last = p.iloc[-1]
    assert math.isnan(last["bigram_surp_back"]) and math.isnan(p.iloc[-2]["trigram_surp_back"])
    assert math.isnan(p.iloc[0]["bigram_surp_cross"])


def test_missing_word_breaks_cross_context(tmp_path, lexicon):
    # zorpleth has no phones: "sat" must not be scored against "cat"
    p = _run(tmp_path, lexicon, "cat zorpleth sat", condition="nonword").phones
    s = p[p["word_idx"] == 3].iloc[0]
    assert math.isnan(s["bigram_surp_cross"])
    t = p[(p["word_idx"] == 1) & (p["phone_in_word"] == 3)].iloc[0]
    assert math.isnan(t["bigram_surp_back"])


def test_phone_metrics_stop_at_word(tmp_path, lexicon):
    t = _run(tmp_path, lexicon, "cat sat")
    assert "bigram_surp_cross" in t.words.columns
    assert not any(c.startswith(("bigram_", "feat_", "blick_w1", "blick_w2", "blick_w3", "blick_w4", "delta_")) for c in t.stimuli.columns)
    assert "blick_word" in t.stimuli.columns                            # word metric -> stimulus


def test_blick_scores_and_windows(tmp_path, lexicon):
    from linguistic.metrics.phonological.blick import blick_string, score
    assert blick_string(("B", "L", "IH", "K")) == "B L IH1 K"
    assert score(("B", "L", "IH", "K")) < score(("B", "N", "IH", "K"))     # blick vs bnick
    t = _run(tmp_path, lexicon, "cat sat")
    p, w = t.phones, t.words
    last = p[(p["word_idx"] == 1) & (p["phone_in_word"] == 3)].iloc[0]
    # a word of <= 4 phones: the 4-phone within-word window at its last phone is the whole word
    assert math.isclose(last["blick_w4_within"], w.loc[0, "blick_word"])
    s = p[(p["word_idx"] == 2) & (p["phone_in_word"] == 1)].iloc[0]
    assert math.isclose(s["blick_w2_cross"], score(("T", "S")))            # window crosses the boundary
    assert math.isnan(s["delta_blick_w2_within"])                         # word-initial


def test_phonetic_features_membership_and_word_share(tmp_path, lexicon):
    from linguistic.metrics.phonological.features import FEATURES
    assert len(FEATURES) == 14
    assert set().union(*FEATURES.values()) == lexicon.symbols          # every ARPAbet phone covered
    t = _run(tmp_path, lexicon, "cat")                                   # K AE T
    p = t.phones.set_index("phone")
    assert p.loc["K", ["feat_dorsal", "feat_plosive", "feat_obstruent", "feat_voiced"]].tolist() == [1, 1, 1, 0]
    assert p.loc["AE", ["feat_front", "feat_low", "feat_syllabic", "feat_voiced", "feat_sonorant"]].tolist() == [1] * 5
    assert math.isclose(t.words.loc[0, "feat_plosive"], 2 / 3)          # K, T of K AE T


def test_context_ordering_and_dictionary(tmp_path, lexicon):
    t = _run(tmp_path, lexicon, "cat sat")
    cols = t.columns
    w = cols[cols["table"] == "word"]
    assert (w["context"].tolist() == sorted(w["context"].tolist(), key=["independent", "dependent"].index))
    within = cols[(cols["column"] == "bigram_surp_within")].set_index("table")["context"]
    assert within.to_dict() == {"phone": "dependent", "word": "dependent"}
    w1 = cols[(cols["column"] == "blick_w1_within")].set_index("table")["context"]
    assert set(w1) == {"independent"}                                    # single-phone BLICK
    metric_cols = [c for c in t.words.columns if c in set(w["column"])]
    assert metric_cols == w["column"].tolist()
