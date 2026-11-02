"""Command line: `linguistic run`, `linguistic prepare`, `linguistic list`."""
from __future__ import annotations

from pathlib import Path

import typer

from .assemble import run_metrics
from .io import load_stimuli
from .manifest import build_manifest, write_manifest
from .normalize import Lexicon, build_units
from . import registry, resources
from .registry import Context, all_metrics, select

app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.command()
def prepare(input: Path = typer.Option(..., "--input", "-i", exists=True),
            out: Path = typer.Option(..., "--out", "-o")):
    """Build the word and phone tables only, and list words that still need hand phones."""
    _run(input, out, metric_spec=[])


@app.command()
def run(input: Path = typer.Option(..., "--input", "-i", exists=True),
        out: Path = typer.Option(..., "--out", "-o"),
        metrics: list[str] = typer.Option(["all"], "--metrics", "-m",
                                          help="'all', family names or metric names; repeatable")):
    """Extract metrics into words.csv, phones.csv, stimuli.csv + manifest.json."""
    _run(input, out, metric_spec=metrics)


@app.command()
def fetch(name: str = typer.Argument(..., help="resource name, or 'all'")):
    """Download and verify an external lexicon/model into $LINGUISTIC_RESOURCES."""
    names = list(resources.RESOURCES) if name == "all" else [name]
    for n in names:
        r = resources.RESOURCES[n]
        if r.manual:
            typer.echo(f"{n}: manual download, see {r.readme}")
            continue
        typer.echo(f"{n}: {resources.fetch(n)}")


@app.command("list")
def list_metrics():
    """Show the registered metrics."""
    ms = all_metrics()
    if not ms:
        typer.echo("no metrics registered yet")
    for m in ms:
        typer.echo(f"{m.family:14s} {m.name:28s} {m.level:6s} "
                   f"conditions={','.join(sorted(m.conditions))} requires={','.join(sorted(m.requires)) or '-'}")


def _run(input: Path, out: Path, metric_spec: list[str]) -> None:
    stimuli = load_stimuli(input)
    lexicon = Lexicon()
    units = build_units(stimuli, lexicon)
    metrics = select(metric_spec) if metric_spec else []
    for name, missing in registry.skipped.items():
        typer.echo(f"skipped {name}: resource {', '.join(missing)} not installed "
                   f"(`linguistic fetch` or see the family README)")
    ctx = Context(stimuli=stimuli, words=units.words, phones=units.phones, lexicon=lexicon)
    tables = run_metrics(ctx, metrics)

    # built before writing, so the code-commit check is not affected by these outputs
    manifest = build_manifest(input, stimuli, metrics, lexicon.info(), tables.report, units)
    manifest["skipped_metrics"] = dict(registry.skipped)
    manifest["resources"] = [resources.info(r) for r in sorted({r for m in metrics for r in m.resources})]
    out.mkdir(parents=True, exist_ok=True)
    tables.words.to_csv(out / "words.csv", index=False)
    tables.phones.to_csv(out / "phones.csv", index=False)
    tables.stimuli.to_csv(out / "stimuli.csv", index=False)
    units.missing_phones.to_csv(out / "missing_phones.csv", index=False)
    write_manifest(manifest, out)

    src = units.words["phone_source"].fillna("missing").value_counts().to_dict()
    typer.echo(f"{len(stimuli)} stimuli, {len(units.words)} words, {len(units.phones)} phones; "
               f"phone sources {src}; {len(metrics)} metric(s) -> {out}")
    if len(units.missing_phones):
        typer.echo(f"{len(units.missing_phones)} word(s) need hand phones: see "
                   f"{out / 'missing_phones.csv'}")


if __name__ == "__main__":
    app()
