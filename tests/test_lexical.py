import math

import pandas as pd
import pytest

from linguistic import registry, resources
from linguistic.assemble import run_metrics
from linguistic.io import load_stimuli
from linguistic.metrics.lexical import frequency
from linguistic.metrics.lexical.cohort import cohort_values
from linguistic.normalize import build_units
from linguistic.registry import Context, select

from conftest import write_labels, write_stimuli


def test_cohort_values_definition():
    pre = {("K",): frozenset({"cat", "cap", "cot"}), ("K", "AE"): frozenset({"cat", "cap"}),
           ("K", "AE", "T"): frozenset({"cat"})}
    v = cohort_values(["K", "AE", "T"], "cat", pre)
    assert math.isclose(v["cohort_density_log10"], math.log10(1 + 2 + 1 + 0))
    assert v["uniqueness_point_idx"] == 3 and v["up_reached"]
    v = cohort_values(["K", "AE"], "ca", pre)            # never unique -> last phoneme
    assert v["uniqueness_point_idx"] == 2 and not v["up_reached"]


@pytest.fixture
def fake_resources(tmp_path, monkeypatch):
    monkeypatch.setenv("LINGUISTIC_RESOURCES", str(tmp_path / "res"))
    (tmp_path / "res" / "subtlex_us").mkdir(parents=True)
    pd.DataFrame({"Word": ["the", "cat", "icecream"], "FREQcount": [1000, 99, 9],
                  "Lg10WF": [3.0004, 2.0, 1.0]}).to_csv(
        tmp_path / "res" / "subtlex_us" / "SUBTLEXus74286wordstextversion.txt", sep="\t", index=False)
    frequency._subtlex.cache_clear()
    frequency._elp.cache_clear()
    yield tmp_path / "res"
    frequency._subtlex.cache_clear()
    frequency._elp.cache_clear()


def _run(tmp_path, lexicon, rows, spec):
    s = load_stimuli(write_stimuli(tmp_path / "s.csv", rows))
    u = build_units(s, lexicon)
    return run_metrics(Context(s, u.words, u.phones, lexicon), select(spec))


def test_subtlex_sentence_only_and_closed_form(tmp_path, lexicon, fake_resources):
    t = _run(tmp_path, lexicon, [
        {"stimulus_id": "s", "condition": "sentence", "text": "the cat ice-cream zorp"},
        {"stimulus_id": "n", "condition": "nonword", "text": "cat"}], ["subtlex_frequency"])
    w = t.words.set_index(["stimulus_id", "word"])
    assert w.loc[("s", "cat"), "subtlex_lg10wf"] == 2.0
    assert math.isclose(w.loc[("s", "cat"), "subtlex_zipf"], math.log10(100 / 51.074286) + 3)
    assert w.loc[("s", "ice-cream"), "subtlex_lg10wf"] == 1.0          # closed-form lookup
    assert math.isnan(w.loc[("s", "zorp"), "subtlex_lg10wf"])
    assert math.isnan(w.loc[("n", "cat"), "subtlex_lg10wf"])          # nonword condition


def test_missing_resource_skipped_via_family_but_not_by_name(fake_resources):
    names = [m.name for m in select(["lexical"])]
    assert "elp_frequency" not in names and registry.skipped == {"elp_frequency": ["elp"]}
    assert [m.name for m in select(["elp_frequency"])] == ["elp_frequency"]


def test_elp_reads_user_export(tmp_path, lexicon, fake_resources):
    (fake_resources / "elp").mkdir()
    (fake_resources / "elp" / "elp_words.csv").write_text("Word,Log_Freq_HAL\ncat,9.5\nthe,#\n")
    t = _run(tmp_path, lexicon, [{"stimulus_id": "s", "condition": "sentence", "text": "the cat"}],
             ["elp_frequency"])
    assert t.words["elp_log_freq_hal"].tolist()[1] == 9.5
    assert math.isnan(t.words["elp_log_freq_hal"].tolist()[0])         # ELP's '#' = missing


def test_elp_wrong_columns(tmp_path, lexicon, fake_resources):
    (fake_resources / "elp").mkdir()
    (fake_resources / "elp" / "elp_words.csv").write_text("item,freq\ncat,1\n")
    with pytest.raises(ValueError, match="needs columns"):
        _run(tmp_path, lexicon, [{"stimulus_id": "s", "condition": "sentence", "text": "cat"}],
             ["elp_frequency"])


def test_duration_and_uniqueness_ms_need_timing(tmp_path, lexicon):
    write_labels(tmp_path / "w.txt", [(0.0, 0.3, "cat")])
    write_labels(tmp_path / "p.txt", [(0.0, 0.1, "K"), (0.1, 0.2, "AE"), (0.2, 0.3, "T")])
    t = _run(tmp_path, lexicon, [
        {"stimulus_id": "h", "condition": "sentence", "text": "cat",
         "words_labels": "w.txt", "phones_labels": "p.txt"},
        {"stimulus_id": "c", "condition": "sentence", "text": "cat"}], ["cohort", "word_duration"])
    w = t.words.set_index("stimulus_id")
    assert math.isclose(w.loc["h", "word_duration_ms"], 300.0)
    up = int(w.loc["h", "uniqueness_point_idx"])
    assert math.isclose(w.loc["h", "uniqueness_point_ms"], up * 100.0)
    assert math.isnan(w.loc["c", "word_duration_ms"]) and math.isnan(w.loc["c", "uniqueness_point_ms"])
    assert w.loc["c", "cohort_density_log10"] == w.loc["h", "cohort_density_log10"]
