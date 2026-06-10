import json
from pathlib import Path
from typing import Any, Iterable


def write_jsonl_record(f, record: dict[str, Any]) -> None:
    f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_jsonl(
    records: Iterable[dict[str, Any]],
    path: str | Path,
) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    count = 0

    with path.open("w", encoding="utf-8") as f:
        for record in records:
            write_jsonl_record(f, record)
            count += 1

    return count


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)

    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]
