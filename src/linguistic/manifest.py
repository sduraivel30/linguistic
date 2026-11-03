"""Provenance written next to every run's tables."""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

from . import __version__

_PACKAGES = ("numpy", "pandas", "nltk", "typer")


def sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_manifest(input_path, stimuli, metrics, lexicon_info, report, units) -> dict:
    base = Path(input_path).resolve().parent
    label_files = sorted({p for col in ("words_labels", "phones_labels", "labels")
                          for p in stimuli[col] if p})
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "linguistic_version": __version__,
        "linguistic_git_commit": _git_commit(),
        "python": platform.python_version(),
        "packages": {p: _version(p) for p in _PACKAGES},
        "input": {"path": str(input_path), "sha256": sha256(input_path),
                  "n_stimuli": int(len(stimuli)),
                  "n_by_condition": stimuli["condition"].value_counts().to_dict()},
        "label_files": {_rel(p, base): sha256(p) for p in label_files},
        "lexicon": lexicon_info,
        "phone_sources": units.words["phone_source"].fillna("missing").value_counts().to_dict(),
        "n_words_missing_phones": int(len(units.missing_phones)),
        "metrics": [m.info() for m in metrics],
        "report": report,
    }


def _rel(path: str, base: Path) -> str:
    """Path relative to the stimulus file's folder when possible, else the file name."""
    p = Path(path).resolve()
    try:
        return str(p.relative_to(base))
    except ValueError:
        return p.name


def write_manifest(manifest: dict, out_dir: str | Path) -> Path:
    path = Path(out_dir) / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2, default=str))
    return path


def _version(pkg: str) -> str | None:
    try:
        return metadata.version(pkg)
    except metadata.PackageNotFoundError:
        return None


def _git_commit() -> str | None:
    try:
        root = Path(__file__).resolve().parents[2]
        out = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                             capture_output=True, text=True, check=True)
        # dirty = uncommitted changes to the code itself (outputs and data don't count)
        dirty = subprocess.run(["git", "-C", str(root), "status", "--porcelain",
                                "--untracked-files=no", "--", "src", "pyproject.toml"],
                               capture_output=True, text=True, check=True).stdout.strip()
        return out.stdout.strip() + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return None
