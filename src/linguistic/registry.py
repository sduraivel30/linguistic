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
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .normalize import Lexicon

LEVELS = {"word": ("stimulus_id", "word_idx"),
          "phone": ("stimulus_id", "word_idx", "phone_in_word")}
REQUIREMENTS = {"phones", "timing", "audio"}


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

    def compute(self, ctx: Context) -> pd.DataFrame:
        raise NotImplementedError

    def info(self) -> dict:
        return {"name": self.name, "family": self.family, "level": self.level,
                "columns": list(self.columns), "conditions": sorted(self.conditions),
                "requires": sorted(self.requires), "description": self.description,
                "nan_reason": self.nan_reason}


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
    """Resolve 'all', family names and metric names to metrics (registry order, no duplicates)."""
    _load_builtin()
    families = {m.family for m in _REGISTRY.values()}
    chosen: set[str] = set()
    for item in spec:
        if item == "all":
            chosen |= set(_REGISTRY)
        elif item in _REGISTRY:
            chosen.add(item)
        elif item in families:
            chosen |= {n for n, m in _REGISTRY.items() if m.family == item}
        else:
            raise KeyError(f"unknown metric or family '{item}'; "
                           f"known: {sorted(_REGISTRY)} / families {sorted(families)}")
    return [m for n, m in _REGISTRY.items() if n in chosen]


def _load_builtin() -> None:
    from . import metrics  # noqa: F401  (imports register the built-in metrics)
