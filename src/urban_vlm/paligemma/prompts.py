from typing import Any

import pandas as pd

from urban_vlm.eubucco.schema import EubuccoField
from urban_vlm.paligemma.config import PaliGemmaTask


def build_prompt(record: dict[str, Any], task: PaliGemmaTask | str) -> str:
    task = PaliGemmaTask(task)

    if task == PaliGemmaTask.BUILDING_YEAR:
        return (
            "Predict the construction year of the building in this aerial image crop. "
            "Answer with only a four-digit year, or unknown."
        )

    if task == PaliGemmaTask.BUILDING_DECADE:
        return (
            "Predict the construction decade of the building in this aerial image crop. "
            "Answer with only a decade like 1990, or unknown."
        )

    raise ValueError(f"Unsupported PaliGemma task: {task}")


def build_target(record: dict[str, Any], task: PaliGemmaTask | str) -> str:
    task = PaliGemmaTask(task)
    attributes = _first_building(record).get("attributes", {})

    if task == PaliGemmaTask.BUILDING_YEAR:
        return _format_year(attributes.get(str(EubuccoField.construction_year)))

    if task == PaliGemmaTask.BUILDING_DECADE:
        return _format_year(attributes.get("construction_decade"))

    raise ValueError(f"Unsupported PaliGemma task: {task}")


def _first_building(record: dict[str, Any]) -> dict[str, Any]:
    buildings = record.get("buildings") or []

    if not buildings:
        raise ValueError(f"Record has no buildings: {record.get('id')}")

    return buildings[0]


def _format_year(value: Any) -> str:
    if value is None or pd.isna(value):
        return "unknown"

    return str(int(value))
