import math

import pandas as pd
import pytest

from linguistic.io import InputError, load_stimuli
from linguistic.normalize import build_units, tokenize

from conftest import write_labels, write_stimuli


def test_tokenize_strips_edge_punctuation_only():
    assert tokenize("Tom didn't wait, at the bus-stop.") == \
        ["Tom", "didn't", "wait", "at", "the", "bus-stop"]


def _units(tmp_path, rows, lexicon):
    return build_units(load_stimuli(write_stimuli(tmp_path / "s.csv", rows)), lexicon)


def test_cmudict_words_and_missing_nonwords(tmp_path, lexicon):
    u = _units(tmp_path, [
        {"stimulus_id": "s1", "condition": "sentence", "text": "The cat"},
        {"stimulus_id": "n1", "condition": "nonword", "text": "zorpleth cat"},
    ], lexicon)
    w = u.words.set_index(["stimulus_id", "word_idx"])
    assert w.loc[("s1", 2), "phones"] == "K AE T"
    assert w.loc[("s1", 2), "phone_source"] == "cmudict"
    assert bool(w.loc[("n1", 2), "in_lexicon"]) is True          # a nonword list item found in CMUdict
    assert bool(w.loc[("n1", 1), "in_lexicon"]) is False
    assert pd.isna(w.loc[("n1", 1), "phone_source"])
    assert u.missing_phones[["stimulus_id", "word"]].values.tolist() == [["n1", "zorpleth"]]
    # cmudict phones carry no timing
    assert u.phones["start"].isna().all()
    assert u.phones.loc[u.phones["stimulus_id"] == "s1", "phone_idx"].tolist() == [1, 2, 3, 4, 5]


def test_hand_phones_override_and_majority_overlap(tmp_path, lexicon):
    # the DH straddles the word boundary at 0.50 but lies mostly in "the" -> goes to "the"
    write_labels(tmp_path / "w.txt", [(0.00, 0.50, "after"), (0.50, 0.80, "the")])
    write_labels(tmp_path / "p.txt", [
        (0.00, 0.10, "AE1"), (0.10, 0.20, "F"), (0.20, 0.30, "T"), (0.30, 0.49, "ER0"),
        (0.49, 0.60, "DH"), (0.60, 0.80, "AH0"), (0.80, 0.90, "sil")])
    u = _units(tmp_path, [{"stimulus_id": "s1", "condition": "sentence", "text": "after the",
                           "words_labels": "w.txt", "phones_labels": "p.txt"}], lexicon)
    w = u.words.set_index("word")
    assert w.loc["after", "phones"] == "AE F T ER"
    assert w.loc["the", "phones"] == "DH AH"
    assert (w["phone_source"] == "hand").all()
    # word bounds re-derived from phones
    assert math.isclose(w.loc["after", "offset"], 0.49)
    assert math.isclose(w.loc["the", "onset"], 0.49)
    assert len(u.missing_phones) == 0


def test_hand_phones_for_unknown_nonword(tmp_path, lexicon):
    write_labels(tmp_path / "w.txt", [(0.0, 0.4, "zorpleth")])
    write_labels(tmp_path / "p.txt", [(0.0, 0.1, "Z"), (0.1, 0.2, "AO"), (0.2, 0.4, "R")])
    u = _units(tmp_path, [{"stimulus_id": "n1", "condition": "nonword", "text": "zorpleth",
                           "words_labels": "w.txt", "phones_labels": "p.txt"}], lexicon)
    assert u.words.loc[0, "phone_source"] == "hand"
    assert bool(u.words.loc[0, "in_lexicon"]) is False


def test_hyphenated_lookup_routes(lexicon):
    assert lexicon.resolve("cat")[1] == "whole"
    prons, route = lexicon.resolve("bus-stop")
    assert route == "parts"
    assert prons == [lexicon.lookup("bus")[0] + lexicon.lookup("stop")[0]]
    assert lexicon.resolve("zorp-cat") == ([], None)            # one part unknown
    assert lexicon.resolve("zorpleth") == ([], None)


def test_hyphenated_word_split_in_labels_is_merged(tmp_path, lexicon):
    write_labels(tmp_path / "w.txt", [(0.0, 0.3, "ice"), (0.3, 0.7, "cream")])
    write_labels(tmp_path / "p.txt", [(0.0, 0.15, "AY"), (0.15, 0.3, "S"), (0.3, 0.4, "K"),
                                      (0.4, 0.5, "R"), (0.5, 0.6, "IY"), (0.6, 0.7, "M")])
    u = _units(tmp_path, [{"stimulus_id": "s1", "condition": "sentence", "text": "ice-cream",
                           "words_labels": "w.txt", "phones_labels": "p.txt"}], lexicon)
    assert len(u.words) == 1
    assert u.words.loc[0, "phones"] == "AY S K R IY M"
    assert u.words.loc[0, "hand_word_label"] == "ice cream"


@pytest.mark.parametrize("words,phones,msg", [
    ([(0, 1, "after")], [(0, 1, "AE")], "1 words in"),
    ([(0, 1, "after"), (1, 2, "the")], [(0, 1, "æ")], "not ARPAbet"),
    ([(0, 1, "after"), (1, 2, "the")], [(3, 4, "DH")], "overlap no word"),
])
def test_hand_label_errors(tmp_path, lexicon, words, phones, msg):
    write_labels(tmp_path / "w.txt", words)
    write_labels(tmp_path / "p.txt", phones)
    with pytest.raises(InputError, match=msg):
        _units(tmp_path, [{"stimulus_id": "s1", "condition": "sentence", "text": "after the",
                           "words_labels": "w.txt", "phones_labels": "p.txt"}], lexicon)


def test_combined_label_file(tmp_path, lexicon):
    # one file, both tiers (MT style: words lowercase, phones uppercase; silence word dropped)
    (tmp_path / "mt.txt").write_text(
        "0.0\t0.3\tcat\n0.3\t0.4\tsil\n0.4\t0.7\tsat\n"
        "0.0\t0.1\tK\n0.1\t0.2\tAE1\n0.2\t0.3\tT\n0.4\t0.5\tS\n0.5\t0.6\tAE1\n0.6\t0.7\tT\n")
    u = _units(tmp_path, [{"stimulus_id": "s1", "condition": "sentence", "text": "cat sat",
                           "labels": "mt.txt"}], lexicon)
    assert u.words["phones"].tolist() == ["K AE T", "S AE T"]
    assert (u.words["phone_source"] == "hand").all()


def test_all_bad_stimuli_reported_together(tmp_path, lexicon):
    write_labels(tmp_path / "w.txt", [(0, 1, "after")])
    write_labels(tmp_path / "p.txt", [(0, 1, "AE")])
    rows = [{"stimulus_id": f"s{i}", "condition": "sentence", "text": "after the",
             "words_labels": "w.txt", "phones_labels": "p.txt"} for i in (1, 2)]
    with pytest.raises(InputError, match="2 stimulus label problem"):
        _units(tmp_path, rows, lexicon)
