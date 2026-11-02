import pytest

from linguistic.io import InputError, load_stimuli, read_audacity_labels

from conftest import write_labels, write_stimuli


def test_load_stimuli_valid_and_relative_paths(tmp_path):
    write_labels(tmp_path / "w.txt", [(0, 1, "cat")])
    write_labels(tmp_path / "p.txt", [(0, 1, "K")])
    p = write_stimuli(tmp_path / "s.csv", [
        {"stimulus_id": "a", "condition": "Sentence", "text": "the cat",
         "words_labels": "w.txt", "phones_labels": "p.txt"},
        {"stimulus_id": "b", "condition": "nonword", "text": "blick",
         "words_labels": "", "phones_labels": ""},
    ])
    df = load_stimuli(p)
    assert list(df["condition"]) == ["sentence", "nonword"]
    assert df.loc[0, "words_labels"] == str(tmp_path / "w.txt")
    assert df.loc[1, "audio_path"] == ""


@pytest.mark.parametrize("rows,msg", [
    ([{"stimulus_id": "a", "condition": "sentence"}], "missing required"),
    ([{"stimulus_id": "a", "condition": "sentence", "text": "x"},
      {"stimulus_id": "a", "condition": "sentence", "text": "y"}], "duplicate"),
    ([{"stimulus_id": "a", "condition": "word", "text": "x"}], "condition must be"),
    ([{"stimulus_id": "a", "condition": "sentence", "text": " "}], "empty text"),
    ([{"stimulus_id": "a", "condition": "sentence", "text": "x", "words_labels": "w.txt"}],
     "must be given together"),
])
def test_load_stimuli_rejects(tmp_path, rows, msg):
    with pytest.raises(InputError, match=msg):
        load_stimuli(write_stimuli(tmp_path / "s.csv", rows))


def test_read_audacity_labels_sorts_and_skips(tmp_path):
    p = tmp_path / "l.txt"
    p.write_text("0.5\t0.9\tB\n\\\t100\t200\n\n0.0\t0.5\tA\n")
    assert [r["label"] for r in read_audacity_labels(p)] == ["A", "B"]


def test_read_audacity_labels_rejects_bad_line(tmp_path):
    p = tmp_path / "l.txt"
    p.write_text("0.5 0.9 B\n")
    with pytest.raises(InputError, match="start<TAB>end"):
        read_audacity_labels(p)
