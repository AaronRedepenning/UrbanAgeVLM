from typing import Any

import numpy as np
import pandas as pd

from urban_vlm.eubucco.schema import EubuccoField
from urban_vlm.paligemma.config import PaliGemmaTask

BUILDING_CLASSES = {
    "before 1800": (-np.inf, 1799),
    "1800-1850": (1800, 1850),
    "1851-1900": (1851, 1900),
    "1901-1950": (1901, 1950),
    "1951-2000": (1951, 2000),
    "2001-2015": (2001, 2015),
    "after 2015": (2016, np.inf),
}


def build_prompt(
    record: dict[str, Any],
    task: PaliGemmaTask | str,
    lan: str = "en",
) -> str:
    task = PaliGemmaTask(task)

    if lan != "en":
        raise ValueError(f"Unsupported language: {lan}")

    prompts = {
        PaliGemmaTask.BUILDING_YEAR: (
            "what year was the centered building constructed? answer with a four-digit year."
        ),
        PaliGemmaTask.BUILDING_DECADE: (
            "what decade was the centered building constructed in? answer with a decade such as 1970."
        ),
        PaliGemmaTask.BUILDING_CLASS: (
            f"what construction period is the centered building from? choose one: {", ".join(BUILDING_CLASSES.keys())}."
        ),
    }

    try:
        prompt = prompts[task]
    except KeyError as exc:
        raise ValueError(f"Unsupported PaliGemma task: {task}") from exc

    return f"<image>answer {lan} {prompt}\n"


def build_target(record: dict[str, Any], task: PaliGemmaTask | str) -> str:
    task = PaliGemmaTask(task)
    attributes = _first_building(record).get("attributes", {})

    if task == PaliGemmaTask.BUILDING_YEAR:
        return _format_year(attributes.get(str(EubuccoField.construction_year)))

    if task == PaliGemmaTask.BUILDING_DECADE:
        return _format_year(attributes.get("construction_decade"))

    if task == PaliGemmaTask.BUILDING_CLASS:
        return _get_building_class(attributes.get("construction_decade"))

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


def _get_building_class(value: Any) -> str:
    if value is None or pd.isna(value):
        return "unknown"

    value = int(value)

    for key, (start, end) in BUILDING_CLASSES.items():
        if value <= end:
            return key

    return "unknown"
