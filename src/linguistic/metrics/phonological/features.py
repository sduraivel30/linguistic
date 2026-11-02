"""Phonetic feature membership (phone level, context-independent).

14 binary features in the standard intracranial-speech convention (Mesgarani et al. 2014,
Science 343:1006-1010). Memberships follow the TIMIT feature table published at
https://github.com/ChangLabUcsf/HamiltonOganianChang2020 (util1/Timit_features.mat,
BSD-3-Clause), restricted to the 39 ARPAbet phones used here. This module is our own
code; the table is written out below and checked in the tests.

A phone belongs to every feature it is listed under (T: coronal, plosive, obstruent).
Some phones have no place feature (TH, DH, L, R, W, Y, HH): that is the convention.
"""
from __future__ import annotations

import pandas as pd

from ...registry import Metric, register

FEATURES: dict[str, frozenset[str]] = {k: frozenset(v.split()) for k, v in {
    # place
    "dorsal":    "CH G JH K NG SH",
    "coronal":   "CH D JH N S T Z",
    "labial":    "B F M P V",
    # vowel position
    "high":      "EH EY IH IY UH UW",
    "front":     "AE EH EY IH IY",
    "low":       "AA AE AH AO AW AY",
    "back":      "AA AH AO OW UH UW",
    # manner
    "plosive":   "B D G K P T",
    "fricative": "CH DH F HH JH S SH TH V Z ZH",
    "syllabic":  "AA AE AH AO AW AY EH ER EY IH IY OW OY UH UW",
    "nasal":     "M N NG",
    # voicing / major class
    "voiced":    "AA AE AH AO AW AY B D DH EH EY G IH IY L M N NG OW OY R UH UW V W Y Z",
    "obstruent": "B CH D DH F G HH JH K P S SH T TH V Z ZH",
    "sonorant":  "AA AE AH AO AW AY EH EY IH IY L M N NG OW OY R UH UW W Y",
}.items()}


@register
class PhoneticFeatures(Metric):
    name = "phonetic_features"
    family = "phonological"
    level = "phone"
    requires = frozenset({"phones"})
    context = {"phone": "independent", "word": "independent"}
    columns = tuple(f"feat_{f}" for f in FEATURES)
    description = ("1 if the phone belongs to the feature, else 0 (14 features: place, vowel "
                   "position, manner, voicing/major class). Word level: share of the word's "
                   "phones that have the feature.")

    def compute(self, ctx):
        out = ctx.phones[["stimulus_id", "word_idx", "phone_in_word"]].copy()
        for f, members in FEATURES.items():
            out[f"feat_{f}"] = ctx.phones["phone"].isin(members).astype(float)
        return out
