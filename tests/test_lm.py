"""LM metric tests with GPT-2 small from the local Hugging Face cache (skipped if absent)."""
import math
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

from linguistic import registry, resources  # noqa: E402
from linguistic.assemble import run_metrics  # noqa: E402
from linguistic.io import load_stimuli  # noqa: E402
from linguistic.metrics.lm import causal  # noqa: E402
from linguistic.normalize import build_units  # noqa: E402
from linguistic.registry import Context, select  # noqa: E402

from conftest import write_stimuli  # noqa: E402

SNAP = next(iter(sorted((Path.home() / ".cache/huggingface/hub/models--gpt2/snapshots").glob("*/model.safetensors"))), None)
pytestmark = pytest.mark.skipif(SNAP is None, reason="GPT-2 small not in the local HF cache")


@pytest.fixture(scope="module")
def gpt2_small(tmp_path_factory):
    root = tmp_path_factory.mktemp("res")
    (root / "gpt2-test").symlink_to(SNAP.parent)
    mp = pytest.MonkeyPatch()
    mp.setenv("LINGUISTIC_RESOURCES", str(root))
    mp.setenv("LINGUISTIC_DEVICE", "cpu")
    causal._make("gpt2-test", "openai-community/gpt2", "0" * 40, "gpt2test")
    yield causal.load(str(SNAP.parent))
    registry.unregister("lm_gpt2-test")
    resources.RESOURCES.pop("gpt2-test")
    mp.undo()


def test_score_matches_brute_force(gpt2_small):
    import torch
    import torch.nn.functional as F
    tok, model = gpt2_small
    words = ["The", "boy", "unrecognizably", "left"]
    v = causal.score(words, tok, model)
    ids, spans = causal.tokenize(words, tok)
    with torch.no_grad():
        lp = F.log_softmax(model(torch.tensor([ids])).logits[0], -1)
    s, e = spans[2]
    assert v.loc[2, "n_subtokens"] == e - s > 1
    expect = sum(-lp[p - 1, ids[p]].item() for p in range(s, e)) / (e - s)
    assert math.isclose(v.loc[2, "surprisal"], expect, rel_tol=1e-5)
    assert math.isclose(v.loc[2, "entropy"], float(-(lp[s - 1].exp() * lp[s - 1]).sum()), rel_tol=1e-5)
    assert ids[0] == tok.eos_token_id                                  # GPT-2: EOS doubles as BOS


def test_metric_runs_both_conditions(gpt2_small, tmp_path, lexicon):
    s = load_stimuli(write_stimuli(tmp_path / "s.csv", [
        {"stimulus_id": "s", "condition": "sentence", "text": "The cat sat."},
        {"stimulus_id": "n", "condition": "nonword", "text": "blick zorpleth"}]))
    u = build_units(s, lexicon)
    t = run_metrics(Context(s, u.words, u.phones, lexicon), select(["lm_gpt2-test"]))
    assert t.words["lm_surprisal_gpt2test"].notna().all()
    _, spans = causal.tokenize(["blick", "zorpleth"], gpt2_small[0])
    assert t.words["lm_n_subtokens_gpt2test"].tolist()[3:] == [e - s for s, e in spans]
    st = t.stimuli.set_index("stimulus_id")
    assert math.isclose(st.loc["s", "lm_surprisal_gpt2test"],
                        t.words.loc[t.words.stimulus_id == "s", "lm_surprisal_gpt2test"].mean())


def test_missing_model_is_skipped_under_all(monkeypatch, tmp_path):
    monkeypatch.setenv("LINGUISTIC_RESOURCES", str(tmp_path))
    assert "lm_gpt2-xl" not in [m.name for m in select(["lm"])]
    assert registry.skipped["lm_gpt2-xl"] == ["gpt2-xl"]
