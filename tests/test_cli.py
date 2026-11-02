import json

import pandas as pd
from typer.testing import CliRunner

from linguistic.cli import app

from conftest import write_stimuli


def test_prepare_writes_tables_and_manifest(tmp_path):
    inp = write_stimuli(tmp_path / "s.csv", [
        {"stimulus_id": "s1", "condition": "sentence", "text": "The cat sat."},
        {"stimulus_id": "n1", "condition": "nonword", "text": "zorpleth blick"},
    ])
    out = tmp_path / "out"
    res = CliRunner().invoke(app, ["prepare", "-i", str(inp), "-o", str(out)])
    assert res.exit_code == 0, res.output
    for f in ("words.csv", "phones.csv", "stimuli.csv", "missing_phones.csv", "manifest.json"):
        assert (out / f).exists()
    assert pd.read_csv(out / "stimuli.csv")["n_words"].tolist() == [3, 2]
    m = json.loads((out / "manifest.json").read_text())
    assert m["input"]["n_stimuli"] == 2
    assert m["lexicon"]["name"] == "CMUdict (nltk)"
    assert "need hand phones" in res.output
