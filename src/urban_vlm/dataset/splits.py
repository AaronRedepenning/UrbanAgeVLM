import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from urban_vlm.dataset.jsonl import read_jsonl, write_jsonl


def split_jsonl_by_group(
    *,
    input_jsonl: str | Path,
    train_jsonl: str | Path,
    val_jsonl: str | Path,
    test_jsonl: str | Path,
    group_key: str = "image",
    train_fraction: float = 0.8,
    val_fraction: float = 0.1,
    test_fraction: float = 0.1,
    seed: int = 42,
) -> dict[str, Any]:
    input_jsonl = Path(input_jsonl)
    train_jsonl = Path(train_jsonl)
    val_jsonl = Path(val_jsonl)
    test_jsonl = Path(test_jsonl)

    records = read_jsonl(input_jsonl)

    grouped_records: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for record in records:
        if group_key not in record:
            raise KeyError(
                f"Record {record.get('id')} is missing split group key: {group_key}"
            )

        group_value = str(record[group_key])
        grouped_records[group_value].append(record)

    groups = list(grouped_records.keys())

    rng = random.Random(seed)
    rng.shuffle(groups)

    n_groups = len(groups)
    n_train = int(n_groups * train_fraction)
    n_val = int(n_groups * val_fraction)

    train_groups = set(groups[:n_train])
    val_groups = set(groups[n_train : n_train + n_val])
    test_groups = set(groups[n_train + n_val :])

    split_records = {
        "train": [],
        "val": [],
        "test": [],
    }

    for group, group_records in grouped_records.items():
        if group in train_groups:
            split_records["train"].extend(group_records)
        elif group in val_groups:
            split_records["val"].extend(group_records)
        elif group in test_groups:
            split_records["test"].extend(group_records)
        else:
            raise RuntimeError(f"Group was not assigned to any split: {group}")

    write_jsonl(split_records["train"], train_jsonl)
    write_jsonl(split_records["val"], val_jsonl)
    write_jsonl(split_records["test"], test_jsonl)

    return {
        "group_key": group_key,
        "num_records": len(records),
        "num_groups": n_groups,
        "splits": {
            "train": {
                "num_groups": len(train_groups),
                "num_records": len(split_records["train"]),
                "path": str(train_jsonl),
            },
            "val": {
                "num_groups": len(val_groups),
                "num_records": len(split_records["val"]),
                "path": str(val_jsonl),
            },
            "test": {
                "num_groups": len(test_groups),
                "num_records": len(split_records["test"]),
                "path": str(test_jsonl),
            },
        },
    }
