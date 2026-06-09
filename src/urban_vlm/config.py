from pathlib import Path

import yaml


def load_config(path: Path | str) -> dict:
    path = Path(path)
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)
