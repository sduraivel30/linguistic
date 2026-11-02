"""External lexicons and models: where they live, how they are fetched and verified.

Root: $LINGUISTIC_RESOURCES, default ~/.cache/linguistic. Each resource gets its own folder.
`linguistic fetch <name>` downloads and verifies the ones that have a public URL; the others
(manual=True) must be placed by hand, as described in the metric family's README.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Resource:
    name: str
    filename: str              # file the metrics read, inside root/<name>/
    description: str
    url: str = ""              # empty -> manual
    archive_md5: str = ""
    archive_sha256: str = ""
    file_md5: str = ""
    readme: str = ""           # README that documents it
    hf_repo: str = ""          # Hugging Face model: repo id + pinned revision
    hf_revision: str = ""
    hf_patterns: tuple[str, ...] = ("*.json", "*.txt", "*.model", "model*.safetensors")
    container: str = ""        # container image (pulled with apptainer)
    convert: str = ""          # post-download conversion ("glove")

    @property
    def manual(self) -> bool:
        return not (self.url or self.hf_repo or self.container)


RESOURCES = {
    "subtlex_us": Resource(
        name="subtlex_us",
        filename="SUBTLEXus74286wordstextversion.txt",
        description="SUBTLEX-US word frequencies (Brysbaert & New 2009), 74,286 words",
        url="https://www.ugent.be/pp/experimentele-psychologie/en/research/documents/subtlexus/subtlexus2.zip",
        archive_md5="7644415a9f78270c7c3b6b4b6395967c",
        file_md5="ee7be8db21f26e1925ac0ea3ab598f32",
        readme="src/linguistic/metrics/lexical/README.md",
    ),
    "lmzoo-ngram": Resource(
        name="lmzoo-ngram",
        filename="ngram.sif",
        description="lm-zoo SRILM 5-gram image (cpllab/language-models:ngram), run with apptainer",
        container="docker://cpllab/language-models@sha256:cbecfb418878930a06514c2a55e66f116ba56dbc03ece722f73b84fea725df6a",
        readme="src/linguistic/metrics/lm/README.md",
    ),
    "glove-840b": Resource(
        name="glove-840b",
        filename="vectors.npy",
        description="GloVe 840B cased 300-d word vectors (Pennington et al. 2014)",
        # Stanford's official copy on Hugging Face (nlp.stanford.edu is not reachable from some clusters)
        url="https://huggingface.co/stanfordnlp/glove/resolve/1db2080b2d94def6e5b0386a523102f9d8849e9d/glove.840B.300d.zip",
        archive_sha256="f47355dd5b267bd10f08671e513790690233c76a9ffd73aa915d78f894a8912e",
        convert="glove",
        readme="src/linguistic/metrics/lm/README.md",
    ),
    "elp": Resource(
        name="elp",
        filename="elp_words.csv",
        description="English Lexicon Project export (Balota et al. 2007) with a Word and a "
                    "Log_Freq_HAL column; downloaded by hand",
        readme="src/linguistic/metrics/lexical/README.md",
    ),
}


class ResourceMissing(FileNotFoundError):
    pass


def root() -> Path:
    return Path(os.environ.get("LINGUISTIC_RESOURCES", "~/.cache/linguistic")).expanduser()


def path(name: str) -> Path:
    r = RESOURCES[name]
    p = root() / r.name / r.filename
    if not p.exists():
        how = (f"place it there by hand (see {r.readme})" if r.manual
               else f"run `linguistic fetch {r.name}`")
        raise ResourceMissing(f"{r.description}: {p} not found; {how}")
    return p


def md5(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(name: str) -> Path:
    r = RESOURCES[name]
    if r.manual:
        raise ResourceMissing(f"{r.name} has no public download URL; see {r.readme}")
    d = root() / r.name
    d.mkdir(parents=True, exist_ok=True)
    if r.container:
        exe = shutil.which("apptainer") or shutil.which("singularity")
        if not exe:
            raise ResourceMissing(f"{r.name}: apptainer (or singularity) is needed to pull {r.container}")
        import subprocess
        subprocess.run([exe, "pull", "--force", str(d / r.filename), r.container], check=True)
        return d / r.filename
    if r.hf_repo:
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id=r.hf_repo, revision=r.hf_revision, local_dir=d,
                          allow_patterns=list(r.hf_patterns))
        return d / r.filename
    target = d / r.filename
    if target.exists() and (not r.file_md5 or md5(target) == r.file_md5):
        return target
    archive = d / Path(r.url).name
    with urllib.request.urlopen(r.url) as resp, open(archive, "wb") as f:
        shutil.copyfileobj(resp, f)
    if r.archive_md5 and md5(archive) != r.archive_md5:
        raise ValueError(f"{archive}: md5 {md5(archive)} != expected {r.archive_md5}")
    if r.archive_sha256:
        from .manifest import sha256
        if sha256(archive) != r.archive_sha256:
            raise ValueError(f"{archive}: sha256 does not match {r.archive_sha256}")
    if r.convert == "glove":
        from .static_embeddings import convert_glove
        out = convert_glove(archive, d)
        archive.unlink()
        return out
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as z:
            member = next(m for m in z.namelist() if Path(m).name == r.filename)
            with z.open(member) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)
    if r.file_md5 and md5(target) != r.file_md5:
        raise ValueError(f"{target}: md5 {md5(target)} != expected {r.file_md5}")
    return target


def info(name: str) -> dict:
    r = RESOURCES[name]
    p = root() / r.name / r.filename
    out = {"name": r.name, "path": str(p), "url": r.url or "manual"}
    if r.container:
        out.update(url=r.container)
    elif r.hf_repo:
        out.update(url=f"https://huggingface.co/{r.hf_repo}", hf_repo=r.hf_repo,
                   hf_revision=r.hf_revision)
    elif p.exists():
        out["md5"] = md5(p)
    return out


def register_model(name: str, hf_repo: str, revision: str, readme: str,
                   patterns: tuple[str, ...] | None = None) -> None:
    """A Hugging Face model kept at root/<name>/ (config.json + tokenizer + weights)."""
    extra = {"hf_patterns": patterns} if patterns else {}
    RESOURCES[name] = Resource(name=name, filename="config.json",
                               description=f"Hugging Face model {hf_repo}@{revision[:7]}",
                               readme=readme, hf_repo=hf_repo, hf_revision=revision, **extra)
