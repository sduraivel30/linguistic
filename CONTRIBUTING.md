# Contributing

Bug reports and metric requests: open an issue using one of the templates.

To add a metric:

1. Put it in the family folder under `src/linguistic/metrics/<family>/`, subclass
   `linguistic.registry.Metric`, decorate it with `@register`, and import it in the
   family's `__init__.py`.
2. Set its `level` (phone or word), its `context` label at each level (`independent` if
   it depends only on the unit, `dependent` if it depends on a phone or word sequence), and
   any `resources` it needs.
3. Add a test in `tests/` and a row in the family README.

Rules:

- Use established packages; don't reimplement statistics or models.
- Never commit data outputs other than `data/examples/`, and never commit activations,
  embeddings, model weights or containers (`.gitignore` blocks them).
- Pin every external model or lexicon to a revision or checksum in `resources.py` or
  `metrics/lm/models.py`.
- Run `python -m pytest` before opening a pull request.
