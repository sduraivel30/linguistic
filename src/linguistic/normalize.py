"""Turn stimuli into word and phone tables.

Phone source per word, in order of precedence:
  1. 'hand'    -- hand-extracted Audacity labels (words + phones files) for that stimulus
  2. 'cmudict' -- the word's spelling is in CMUdict (first pronunciation, stress stripped)
  3. none      -- phones are missing; phone-based metrics stay NaN and the word is
                  listed in `missing_phones` until hand phones are supplied.
No grapheme-to-phoneme prediction is ever used.

Every word, real or nonword, is tagged `in_lexicon` (spelling found in CMUdict) and
`lexicon_route`: 'whole', or for hyphenated words 'closed' ('icecream') or 'parts'
(part pronunciations joined). Lexical norms such as frequency should accept only
'whole'/'closed' -- a mean over parts is a different quantity.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd

from .io import InputError, read_audacity_labels

_EDGE_PUNCT = re.compile(r"^[^\w']+|[^\w']+$")
_STRESS = re.compile(r"\d")
SILENCE = {"", "SIL", "SP", "SPN", "<SIL>", "<SP>"}


def tokenize(text: str) -> list[str]:
    """Split on whitespace; strip punctuation at word edges (internal ' and - kept)."""
    words = [_EDGE_PUNCT.sub("", t) for t in text.split()]
    return [w for w in words if w]


def word_key(word: str) -> str:
    return word.lower()


def strip_stress(phone: str) -> str:
    return _STRESS.sub("", phone).upper()


class Lexicon:
    """CMUdict pronunciations, stress stripped. Loaded once, lazily."""

    def __init__(self):
        self._d: dict[str, list[tuple[str, ...]]] | None = None
        self._symbols: set[str] | None = None

    def _load(self):
        if self._d is not None:
            return
        import nltk

        try:
            from nltk.corpus import cmudict
            raw = cmudict.dict()
        except LookupError:
            nltk.download("cmudict", quiet=True)
            from nltk.corpus import cmudict
            raw = cmudict.dict()
        self._d = {
            w: [tuple(strip_stress(p) for p in pron) for pron in prons]
            for w, prons in raw.items()
        }
        self._symbols = {p for prons in self._d.values() for pron in prons for p in pron}

    def lookup(self, word: str) -> list[tuple[str, ...]]:
        """Exact-spelling pronunciations (empty if absent)."""
        self._load()
        return self._d.get(word_key(word), [])

    def resolve(self, word: str) -> tuple[list[tuple[str, ...]], str | None]:
        """Pronunciations and the route that found them.

        'whole'  -- exact spelling
        'closed' -- hyphenated word written closed ('ice-cream' -> 'icecream')
        'parts'  -- every hyphen part found; first pronunciations joined in order
        None     -- not found
        """
        prons = self.lookup(word)
        if prons:
            return prons, "whole"
        parts = [p for p in word.split("-") if p]
        if len(parts) < 2:
            return [], None
        prons = self.lookup("".join(parts))
        if prons:
            return prons, "closed"
        part_prons = [self.lookup(p) for p in parts]
        if all(part_prons):
            return [tuple(ph for pp in part_prons for ph in pp[0])], "parts"
        return [], None

    @property
    def symbols(self) -> set[str]:
        self._load()
        return self._symbols

    def info(self) -> dict:
        import nltk

        self._load()
        return {"name": "CMUdict (nltk)", "nltk_version": nltk.__version__,
                "n_entries": len(self._d), "stress": "stripped",
                "pronunciation": "first listed"}


@dataclass
class Units:
    words: pd.DataFrame
    phones: pd.DataFrame
    missing_phones: pd.DataFrame


def build_units(stimuli: pd.DataFrame, lexicon: Lexicon | None = None) -> Units:
    lexicon = lexicon or Lexicon()
    word_rows, phone_rows = [], []
    for s in stimuli.itertuples(index=False):
        tokens = tokenize(s.text)
        hand = _hand_words(s, tokens, lexicon) if s.phones_labels else None
        for i, w in enumerate(tokens):
            prons, route = lexicon.resolve(w)
            row = {
                "stimulus_id": s.stimulus_id, "condition": s.condition,
                "word_idx": i + 1, "word": w, "word_key": word_key(w),
                "in_lexicon": bool(prons), "lexicon_route": route,
                "n_pronunciations": len(prons),
                "phone_source": None, "onset": float("nan"), "offset": float("nan"),
                "hand_word_label": None,
            }
            phones: list[dict] = []
            if hand is not None:
                h = hand[i]
                row.update(onset=h["start"], offset=h["end"], hand_word_label=h["label"])
                if h["phones"]:
                    phones, row["phone_source"] = h["phones"], "hand"
            if not phones and prons:
                phones = [{"phone": p, "start": float("nan"), "end": float("nan")} for p in prons[0]]
                row["phone_source"] = "cmudict"
            row["phones"] = " ".join(p["phone"] for p in phones)
            row["n_phones"] = len(phones)
            word_rows.append(row)
            for j, p in enumerate(phones):
                phone_rows.append({
                    "stimulus_id": s.stimulus_id, "condition": s.condition,
                    "word_idx": i + 1, "phone_in_word": j + 1,
                    "phone": p["phone"], "start": p["start"], "end": p["end"],
                })

    words = pd.DataFrame(word_rows)
    phones = pd.DataFrame(phone_rows, columns=[
        "stimulus_id", "condition", "word_idx", "phone_in_word", "phone", "start", "end"])
    if len(phones):
        phones["phone_idx"] = phones.groupby("stimulus_id").cumcount() + 1
    else:
        phones["phone_idx"] = pd.Series(dtype=int)
    missing = words.loc[words["phone_source"].isna(),
                        ["stimulus_id", "condition", "word_idx", "word"]].reset_index(drop=True)
    return Units(words=words, phones=phones, missing_phones=missing)


def _hand_words(s, tokens: list[str], lexicon: Lexicon) -> list[dict]:
    """Hand label words with their phones, assigned by greatest time overlap.

    Word bounds are then re-derived from the phones (phone timing is ground truth).
    """
    words = _merge_hyphen_parts(
        [w for w in read_audacity_labels(s.words_labels) if w["label"].strip()], tokens)
    if len(words) != len(tokens):
        raise InputError(
            f"{s.stimulus_id}: {len(words)} words in {s.words_labels} "
            f"but {len(tokens)} in text {tokens}"
        )
    phones = []
    for p in read_audacity_labels(s.phones_labels):
        label = strip_stress(p["label"])
        if label in SILENCE:
            continue
        if label not in lexicon.symbols:
            raise InputError(
                f"{s.stimulus_id}: phone '{p['label']}' in {s.phones_labels} is not ARPAbet"
            )
        phones.append({**p, "phone": label})

    out = [{**w, "phones": []} for w in words]
    orphans = []
    for p in phones:
        overlaps = [min(p["end"], w["end"]) - max(p["start"], w["start"]) for w in words]
        best = max(range(len(words)), key=lambda k: overlaps[k])
        if overlaps[best] <= 0:
            orphans.append(f"{p['phone']}@{p['start']:.3f}")
            continue
        out[best]["phones"].append({k: p[k] for k in ("phone", "start", "end")})
    if orphans:
        raise InputError(
            f"{s.stimulus_id}: phones overlap no word interval: {orphans}"
        )
    for w in out:
        if w["phones"]:
            w["start"], w["end"] = w["phones"][0]["start"], w["phones"][-1]["end"]
    return out


def _merge_hyphen_parts(labels: list[dict], tokens: list[str]) -> list[dict]:
    """Merge consecutive label words that spell the parts of a hyphenated text word.

    e.g. text 'ice-cream' with label words 'ice', 'cream' -> one 'ice-cream' interval.
    Labels that don't line up this way are returned unchanged (the count check then fails).
    """
    out, j = [], 0
    for t in tokens:
        parts = [word_key(p) for p in t.split("-") if p]
        span = labels[j:j + len(parts)]
        if (len(parts) > 1 and len(span) == len(parts)
                and [word_key(_EDGE_PUNCT.sub("", w["label"])) for w in span] == parts):
            out.append({"start": span[0]["start"], "end": span[-1]["end"],
                        "label": " ".join(w["label"] for w in span)})
            j += len(parts)
        elif j < len(labels):
            out.append(labels[j])
            j += 1
    return out + labels[j:]
