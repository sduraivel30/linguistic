import pandas as pd
import pytest

from linguistic.normalize import Lexicon


@pytest.fixture(scope="session")
def lexicon():
    return Lexicon()


def write_labels(path, rows):
    path.write_text("".join(f"{s:.4f}\t{e:.4f}\t{lab}\n" for s, e, lab in rows))
    return path


def write_stimuli(path, rows):
    pd.DataFrame(rows).to_csv(path, index=False)
    return path
