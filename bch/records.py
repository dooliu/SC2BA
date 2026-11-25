"""Write and verify experiment records."""

from hashlib import sha256
import json
from pathlib import Path
from tempfile import NamedTemporaryFile

FOLDS = ("train", "validation", "test")


def json_line(record):
    return json.dumps(record, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n"


def write_json(path, record):
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        temp = Path(stream.name)
        stream.write(json_line(record))
    temp.replace(path)


def digest_file(path):
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
