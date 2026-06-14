from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)

    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if raw is None:
        raise ValueError(f"Config file is empty: {path}")

    if not isinstance(raw, dict):
        raise TypeError(f"Config file must contain a YAML mapping: {path}")

    return raw


def url_filename(url: str) -> str:
    return Path(unquote(urlparse(url).path)).name
