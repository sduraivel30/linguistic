"""Metric plugin contract and registry.

A metric subclasses `Metric`, sets its class attributes, implements `compute`,
and is registered with `@register`. Its module must be imported by
`linguistic.metrics` for the registration to happen.

`compute(ctx)` returns a DataFrame with the key columns for its level plus one
column per name in `columns`:
    level 'word'  -> keys stimulus_id, word_idx
    level 'phone' -> keys stimulus_id, word_idx, phone_in_word
Rows may be omitted; missing values become NaN. The runner (assemble.py)
enforces `conditions` and `requires`, so `compute` does not need to.

Context, per level. At each level a column is
    'independent' -- its value depends only on that unit itself (a phone's identity,
                     a word's spelling/phones/frequency), or
    'dependent'   -- it depends on neighbouring units at that level or on the larger
                     unit the unit sits in (preceding phones, preceding words).
The same column can change class between levels: within-word phoneme surprisal is
dependent for a phone (it uses the preceding phones of the word) but independent for
the word (its per-word mean depends only on the word). `context` gives the class at
level the column appears at: a 'phone' metric also appears at 'word' (mean over the
word's phones), a 'word' metric at 'stimulus' (mean over its words, same class as at
'word'); phone metrics are never carried to the stimulus. `context_overrides` sets it per column where one metric mixes classes.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .normalize import Lexicon

LEVELS = {"word": ("stimulus_id", "word_idx"),
          "phone": ("stimulus_id", "word_idx", "phone_in_word")}
REQUIREMENTS = {"phones", "timing", "audio"}
CONTEXTS = ("independent", "dependent")


@dataclass
class Context:
    stimuli: pd.DataFrame
    words: pd.DataFrame
    phones: pd.DataFrame
    lexicon: Lexicon


class Metric:
    name: str = ""
    family: str = ""
    level: str = "word"
    columns: tuple[str, ...] = ()
    conditions: frozenset[str] = frozenset({"sentence", "nonword"})
    requires: frozenset[str] = frozenset()
    description: str = ""
    nan_reason: str = ""          # why a condition is excluded, documented with the metric
    resources: tuple[str, ...] = ()   # names in linguistic.resources.RESOURCES it needs
    context: dict = {}                # level -> 'independent' | 'dependent' (see module docstring)
    context_overrides: dict = {}      # column -> {level: context}

    def levels(self) -> tuple[str, ...]:
        # carried up one level only: phone -> word (mean over the word's phones),
        # word -> stimulus (mean over the stimulus's words); never phone -> stimulus
        return ("phone", "word") if self.level == "phone" else ("word", "stimulus")

    def column_context(self, column: str, level: str) -> str:
        c = {**self.context, **self.context_overrides.get(column, {})}
        return c.get(level, c["word"]) if level == "stimulus" else c[level]

    def compute(self, ctx: Context) -> pd.DataFrame:
        raise NotImplementedError

    def info(self) -> dict:
        return {"name": self.name, "family": self.family, "level": self.level,
                "columns": list(self.columns), "conditions": sorted(self.conditions),
                "requires": sorted(self.requires), "description": self.description,
                "nan_reason": self.nan_reason, "resources": list(self.resources),
                "context": {c: {lv: self.column_context(c, lv) for lv in self.levels()}
                            for c in self.columns}}


_REGISTRY: dict[str, Metric] = {}


def register(cls: type[Metric]) -> type[Metric]:
    m = cls()
    if not m.name or not m.family:
        raise TypeError(f"{cls.__name__}: name and family are required")
    if m.level not in LEVELS:
        raise TypeError(f"{m.name}: level must be one of {list(LEVELS)}")
    if not m.columns:
        m.columns = (m.name,)
    unknown = set(m.requires) - REQUIREMENTS
    if unknown:
        raise TypeError(f"{m.name}: unknown requirement(s) {unknown}")
    for col in m.columns:
        for lv in m.levels():
            try:
                c = m.column_context(col, lv)
            except KeyError:
                raise TypeError(f"{m.name}: no context given for {col!r} at level {lv!r}") from None
            if c not in CONTEXTS:
                raise TypeError(f"{m.name}: context must be one of {CONTEXTS}, got {c!r}")
    if m.name in _REGISTRY:
        raise TypeError(f"metric '{m.name}' registered twice")
    _REGISTRY[m.name] = m
    return cls


def unregister(name: str) -> None:
    _REGISTRY.pop(name, None)


def all_metrics() -> list[Metric]:
    _load_builtin()
    return list(_REGISTRY.values())


def select(spec: list[str]) -> list[Metric]:
    """Resolve 'all', family names, metric names and '<level>:<context>' selectors
    (e.g. 'phone:independent', 'word:dependent': every metric with a column of that class
    at that level) to metrics (registry order, no duplicates).

    A metric chosen through 'all' or its family is dropped (see `skipped`) when a resource it
    needs is not installed; a metric named explicitly is kept, so its run fails with the
    instructions for getting the resource.
    """
    _load_builtin()
    families = {m.family for m in _REGISTRY.values()}
    chosen: set[str] = set()
    explicit: set[str] = set()
    for item in spec:
        if item == "all":
            chosen |= set(_REGISTRY)
        elif item in _REGISTRY:
            chosen.add(item)
            explicit.add(item)
        elif item in families:
            chosen |= {n for n, m in _REGISTRY.items() if m.family == item}
        elif ":" in item and item.split(":", 1)[1] in CONTEXTS:
            lv, cx = item.split(":", 1)
            chosen |= {n for n, m in _REGISTRY.items() if lv in m.levels()
                       and any(m.column_context(c, lv) == cx for c in m.columns)}
        else:
            raise KeyError(f"unknown metric or family '{item}'; "
                           f"known: {sorted(_REGISTRY)} / families {sorted(families)}")
    skipped.clear()
    out = []
    for n, m in _REGISTRY.items():
        if n not in chosen:
            continue
        missing = [r for r in m.resources if not _installed(r)]
        if missing and n not in explicit:
            skipped[n] = missing
            continue
        out.append(m)
    return out


skipped: dict[str, list[str]] = {}   # metric -> missing resources, from the last select()


def _installed(name: str) -> bool:
    from . import resources
    try:
        resources.path(name)
        return True
    except resources.ResourceMissing:
        return False


def _load_builtin() -> None:
    from . import metrics  # noqa: F401  (imports register the built-in metrics)
