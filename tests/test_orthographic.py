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
