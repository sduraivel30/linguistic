import math

from linguistic.assemble import run_metrics
from linguistic.io import load_stimuli
from linguistic.metrics.orthographic.ngram import V, LetterCounts
from linguistic.normalize import build_units
from linguistic.registry import Context, select

from conftest import write_stimuli


def test_letter_counts_definition():
    c = LetterCounts(["ab", "ac"])
    # bigram 'ab#': P(a|#), P(b|a), P(#|b), add-one over 27 outcomes
    want = -(math.log(3 / (2 + V)) + math.log(2 / (2 + V)) + math.log(2 / (1 + V))) / 3
    assert math.isclose(c.word(2, "ab"), want)
    assert math.isclose(c.word(2, "AB"), want)                  # case-insensitive
    assert c.word(2, "zz") > c.word(2, "ab")                    # unseen -> scored, higher
    assert math.isnan(c.word(2, "'-"))


def test_orthographic_run_and_stimulus_mean(tmp_path, lexicon):
    s = load_stimuli(write_stimuli(tmp_path / "s.csv", [
        {"stimulus_id": "x", "condition": "nonword", "text": "blick bnick"}]))
    u = build_units(s, lexicon)
    t = run_metrics(Context(s, u.words, u.phones, lexicon), select(["orthographic"]))
    w = t.words.set_index("word")["orth_bigram_surp"]
    assert w["bnick"] > w["blick"]
    assert math.isclose(t.stimuli["orth_bigram_surp"].iloc[0], w.mean())


def test_neighbourhood_definitions(monkeypatch):
    import numpy as np
    from linguistic.metrics.orthographic import neighborhood
    from linguistic.metrics.orthographic.neighborhood import BETAS, neighbourhood

    monkeypatch.setattr(neighborhood, "K", 2)

    entries = ["cat", "bat", "cart", "dog", "a"]
    f = np.array([10.0, 5.0, 2.0, 50.0, 100.0])
    v = neighbourhood(["cat", "cot"], entries, f)
    # brute force for "cat": itself (d = 0) left out; d normalised by the longer length
    raw = np.array([np.inf, 1, 1, 3, 2])
    d = raw / np.array([3, 3, 4, 3, 3])
    for b in BETAS:
        w = f * np.exp(-b * d)
        p = w / w.sum()
        assert np.isclose(v[f"entropy_b{b}"][0], -(p[p > 0] * np.log(p[p > 0])).sum())
        assert np.isclose(v[f"surprisal_b{b}"][0], -np.log((f / f.sum() * np.exp(-b * d)).sum()))
    assert v["ld20"][0] == 1.0                                  # mean raw distance to the K = 2 nearest
    # an unrelated string is less supported than a close one
    far = neighbourhood(["cat", "xqzv"], entries, f)
    assert far["surprisal_b20"][1] > far["surprisal_b20"][0]
