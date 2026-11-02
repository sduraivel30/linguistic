"""LM metric and representation tests with small models from the local Hugging Face cache.

GPT-2 small and hf-internal-testing/tiny-random-* models; each test is skipped if its
model is not cached (no downloads during tests).
"""
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

from linguistic import registry, resources  # noqa: E402
from linguistic.assemble import run_metrics  # noqa: E402
from linguistic.io import load_stimuli  # noqa: E402
from linguistic.metrics.lm import bidirectional, causal, models  # noqa: E402
from linguistic.metrics.lm.metric import make_metric  # noqa: E402
from linguistic.normalize import build_units  # noqa: E402
from linguistic.registry import Context, select  # noqa: E402
from linguistic.representations import represent, word_vectors  # noqa: E402

from conftest import write_stimuli  # noqa: E402

HUB = Path.home() / ".cache/huggingface/hub"
CACHED = {
    "t-causal": ("causal", "models--gpt2"),
    "t-masked": ("masked", "models--hf-internal-testing--tiny-random-BertForMaskedLM"),
    "t-albert": ("masked", "models--hf-internal-testing--tiny-random-AlbertForMaskedLM"),
    "t-xlnet": ("xlnet", "models--hf-internal-testing--tiny-random-XLNetLMHeadModel"),
    "t-t5": ("t5", "models--hf-internal-testing--tiny-random-T5ForConditionalGeneration"),
}


def _snapshot(folder):
    snaps = sorted((HUB / folder / "snapshots").glob("*/config.json"))
    return snaps[0].parent if snaps else None


@pytest.fixture(scope="module")
def lm(tmp_path_factory):
    """name -> (tok, model) for every cached test model, registered as metrics too."""
    root = tmp_path_factory.mktemp("res")
    mp = pytest.MonkeyPatch()
    mp.setenv("LINGUISTIC_RESOURCES", str(root))
    mp.setenv("LINGUISTIC_DEVICE", "cpu")
    loaded = {}
    for name, (kind, folder) in CACHED.items():
        snap = _snapshot(folder)
        if snap is None:
            continue
        (root / name).symlink_to(snap)
        spec = models.ModelSpec(name, kind, "test/" + name, "0" * 40, name.replace("-", ""))
        models.add_model(spec)
        register_cls = make_metric(spec)
        registry.register(register_cls)
        tok, model, _ = models.load(name)
        loaded[name] = (tok, model)
    yield loaded
    for name in loaded:
        registry.unregister(f"lm_{name}")
        models.MODELS.pop(name)
        resources.RESOURCES.pop(name)
    models.load.cache_clear()
    mp.undo()


def need(lm, name):
    if name not in lm:
        pytest.skip(f"{name} not in the local HF cache")
    return lm[name]


def test_causal_matches_brute_force(lm):
    import torch
    import torch.nn.functional as F
    tok, model = need(lm, "t-causal")
    words = ["The", "boy", "unrecognizably", "left"]
    v = causal.score(words, tok, model)
    ids, spans = causal.tokenize(words, tok)
    with torch.no_grad():
        lp = F.log_softmax(model(torch.tensor([ids])).logits[0], -1)
    s, e = spans[2]
    expect = sum(-lp[p - 1, ids[p]].item() for p in range(s, e)) / (e - s)
    assert math.isclose(v.loc[2, "surprisal"], expect, rel_tol=1e-5)
    assert math.isclose(v.loc[2, "entropy"], float(-(lp[s - 1].exp() * lp[s - 1]).sum()), rel_tol=1e-5)


def test_masked_pll_whole_word_l2r(lm):
    import torch
    import torch.nn.functional as F
    tok, model = need(lm, "t-masked")
    words = ["the", "unrecognizable", "dog"]
    v = bidirectional.score_masked(words, tok, model)
    enc = tok(words, is_split_into_words=True)
    span = bidirectional.word_spans(enc, 3)[1]
    assert v.loc[1, "n_subtokens"] == len(span) > 1
    nll = []
    for j, p in enumerate(span):                 # mask token j and the word's later tokens
        x = torch.tensor([enc["input_ids"]])
        x[0, span[j:]] = tok.mask_token_id
        with torch.no_grad():
            lp = F.log_softmax(model(input_ids=x).logits[0, p], -1)
        nll.append(-lp[enc["input_ids"][p]].item())
    assert math.isclose(v.loc[1, "surprisal"], float(np.mean(nll)), rel_tol=1e-5)


def test_context_direction(lm):
    """Left-only models ignore later words; bidirectional ones do not."""
    for name, left_only in (("t-xlnet", True), ("t-masked", False), ("t-albert", False)):
        if name not in lm:
            continue
        tok, model = lm[name]
        scorer = {"t-xlnet": bidirectional.score_xlnet, "t-t5": bidirectional.score_t5}.get(
            name, bidirectional.score_masked)
        a = scorer(["the", "cat", "sat"], tok, model)
        b = scorer(["the", "cat", "ran"], tok, model)
        same = a.loc[0, "surprisal"] == b.loc[0, "surprisal"]     # exact: tiny random models move little
        assert same == left_only, name


def test_t5_sees_later_words(lm):
    # the tiny random T5 has logits of ~1e-5, too small for a surprisal difference to survive
    # float32 log-softmax, so compare the logits themselves
    import torch
    tok, model = need(lm, "t-t5")
    y = torch.tensor([tok("<extra_id_0> the <extra_id_1>").input_ids])
    with torch.no_grad():
        a, b = (model(input_ids=torch.tensor([tok(f"<extra_id_0> cat {w}").input_ids]), labels=y).logits
                for w in ("sat", "ran"))
    assert not torch.equal(a, b)


def test_lm_metrics_run(lm, tmp_path, lexicon):
    s = load_stimuli(write_stimuli(tmp_path / "s.csv", [
        {"stimulus_id": "s", "condition": "sentence", "text": "The cat sat."},
        {"stimulus_id": "n", "condition": "nonword", "text": "blick zorpleth"}]))
    u = build_units(s, lexicon)
    names = [f"lm_{n}" for n in lm]
    t = run_metrics(Context(s, u.words, u.phones, lexicon), select(names))
    for n in lm:
        tag = n.replace("-", "")
        assert t.words[f"lm_surprisal_{tag}"].notna().all() and t.words[f"lm_entropy_{tag}"].notna().all()


def test_representations_shapes_and_context(lm, tmp_path, lexicon):
    for name, (tok, model) in lm.items():
        kind = models.MODELS[name].kind
        a, e = word_vectors(["the", "cat", "sat"], tok, model, kind, "incremental", "last")
        n_layers = len(model.encoder.block) if kind == "t5" else model.config.num_hidden_layers
        assert a.shape[:2] == (3, n_layers + 1) and e.shape[0] == 3
        b, e2 = word_vectors(["the", "cat", "ran"], tok, model, kind, "incremental", "last")
        assert np.allclose(a[:2], b[:2], atol=1e-5), name           # incremental: no later words
        assert np.allclose(e[:2], e2[:2]), name                       # static embeddings
        if kind != "causal":
            f, _ = word_vectors(["the", "cat", "sat"], tok, model, kind, "full", "last")
            g, _ = word_vectors(["the", "cat", "ran"], tok, model, kind, "full", "last")
            assert not np.array_equal(f[0], g[0]), name               # full: later words visible


def test_represent_writes_arrays(lm, tmp_path, lexicon):
    name = next(iter(lm), None)
    if name is None:
        pytest.skip("no test model cached")
    p = write_stimuli(tmp_path / "s.csv", [
        {"stimulus_id": "a", "condition": "sentence", "text": "the cat sat"},
        {"stimulus_id": "b", "condition": "sentence", "text": "dogs run"}])
    s = load_stimuli(p)
    u = build_units(s, lexicon)
    out = represent(p, u.words, s, name, tmp_path / "rep", dtype="float16")
    acts = np.load(out / "activations.npy")
    assert acts.shape[0] == 5 and acts.dtype == np.float16
    last = np.load(out / "stimuli_last.npy")
    assert np.array_equal(last[0], acts[2]) and np.array_equal(last[1], acts[4])
    assert (out / "layers.csv").read_text().splitlines()[1].startswith("0,embedding")


def test_missing_model_is_skipped_under_all(monkeypatch, tmp_path):
    monkeypatch.setenv("LINGUISTIC_RESOURCES", str(tmp_path))
    assert "lm_gpt2-xl" not in [m.name for m in select(["lm"])]
    assert registry.skipped["lm_gpt2-xl"] == ["gpt2-xl"]


def test_ngram_align_and_units(monkeypatch, tmp_path, lexicon):
    from linguistic.metrics.lm import ngram
    assert ngram.align(["He", "didn't", "go"], ["he", "did", "n't", "go"]) == [[0], [1, 2], [3]]
    with pytest.raises(AssertionError):
        ngram.align(["cat"], ["dog"])
    sur = pd.DataFrame({"sentence_id": [1] * 5, "token_id": [1, 2, 3, 4, 5],
                        "token": ["he", "did", "n't", "<unk>", "</s>"],
                        "surprisal_bits": [2.0, 4.0, 6.0, 10.0, 1.0]})
    monkeypatch.setattr(ngram, "run_container",
                        lambda sif, lines: ([["he", "did", "n't", "zorp", "</s>"]], sur))
    monkeypatch.setenv("LINGUISTIC_RESOURCES", str(tmp_path))
    (tmp_path / "lmzoo-ngram").mkdir()
    (tmp_path / "lmzoo-ngram" / "ngram.sif").write_text("")
    s = load_stimuli(write_stimuli(tmp_path / "s.csv", [
        {"stimulus_id": "s", "condition": "sentence", "text": "He didn't zorp"},
        {"stimulus_id": "n", "condition": "nonword", "text": "blick"}]))
    u = build_units(s, lexicon)
    w = run_metrics(Context(s, u.words, u.phones, lexicon), select(["ngram_5gram"])).words
    ln2 = math.log(2)
    assert np.allclose(w["ngram_surprisal_5gram"].tolist()[:3], [2 * ln2, 5 * ln2, 10 * ln2])
    assert w["ngram_oov_5gram"].tolist()[:3] == [0, 0, 1]
    assert math.isnan(w["ngram_surprisal_5gram"].tolist()[3])           # nonword not scored


def test_glove_convert_and_represent(tmp_path, monkeypatch, lexicon):
    import zipfile
    from linguistic import static_embeddings as se
    monkeypatch.setattr(se, "DIM", 3)
    z = tmp_path / "g.zip"
    with zipfile.ZipFile(z, "w") as f:
        f.writestr("glove.txt", "the 1 0 0\ncat 0 1 0\n. . 0 0 1\nthe 9 9 9\n")
    res = tmp_path / "res" / "glove-840b"
    res.mkdir(parents=True)
    se.convert_glove(z, res)
    assert (res / "vocab.txt").read_text().split("\n")[:3] == ["the", "cat", ". ."]   # token with a space
    monkeypatch.setenv("LINGUISTIC_RESOURCES", str(tmp_path / "res"))
    se.load_glove.cache_clear()
    vec, found = se.lookup(["the", "CAT", "zorp"])
    assert found == [True, True, False] and vec[0].tolist() == [1, 0, 0]     # duplicate keeps the first
    assert vec[1].tolist() == [0, 1, 0] and np.isnan(vec[2]).all()            # lowercase fallback
    p = write_stimuli(tmp_path / "s.csv", [{"stimulus_id": "a", "condition": "sentence", "text": "the cat zorp"}])
    s = load_stimuli(p)
    out = represent(p, build_units(s, lexicon).words, s, "glove-840b", tmp_path / "rep")
    assert np.allclose(np.load(out / "stimuli_mean.npy")[0], [0.5, 0.5, 0])  # OOV skipped
    se.load_glove.cache_clear()
