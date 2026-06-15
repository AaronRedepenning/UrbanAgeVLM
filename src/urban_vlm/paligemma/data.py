from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from PIL import Image
from rasterio.windows import Window
from torch.utils.data import Dataset

from urban_vlm.dataset.jsonl import read_jsonl


class TargetTask(StrEnum):
    CONSTRUCTION_DECADE = "construction_decade"
    CONSTRUCTION_YEAR = "construction_year"


class JsonlDataset(Dataset):
    def __init__(
        self,
        jsonl_path: str | Path,
        *,
        task: TargetTask,
        max_examples: int | None = None,
    ) -> None:
        self.jsonl_path = Path(jsonl_path)
        self.task = task

        self.records = read_jsonl(self.jsonl_path)
        if max_examples is not None:
            self.records = self.records[:max_examples]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]

        return {
            "id": record["id"],
            "image": load_record_crop(record),
            "prefix": build_prompt(self.task),
            "suffix": get_target(record, self.task),
            "record": record,
        }


def load_record_crop(record: dict[str, Any]) -> Image.Image:
    image_path = record["image"]
    x_min, y_min, x_max, y_max = record["crop"]["bounds"]

    window = Window.from_slices(
        (y_min, y_max),
        (x_min, x_max),
    )

    with rasterio.open(image_path) as src:
        arr = src.read(window=window)

    image = rasterio_array_to_pil(arr)

    return image


def rasterio_array_to_pil(arr: np.ndarray) -> Image.Image:
    """
    Convert rasterio [bands, height, width] to PIL RGB.
    """
    if arr.ndim != 3:
        raise ValueError(f"Expected rasterio array with 3 dims, got shape {arr.shape}")

    if arr.shape[0] >= 3:
        arr = arr[:3]
    elif arr.shape[0] == 1:
        arr = np.repeat(arr, 3, axis=0)
    else:
        raise ValueError(f"Unsupported number of bands: {arr.shape[0]}")

    arr = arr.transpose(1, 2, 0)

    if arr.dtype != np.uint8:
        arr = normalize_to_uint8(arr)

    return Image.fromarray(arr).convert("RGB")


def normalize_to_uint8(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype("float32")

    min_value = float(np.nanmin(arr))
    max_value = float(np.nanmax(arr))

    if max_value <= min_value:
        return np.zeros(arr.shape, dtype=np.uint8)

    arr = (arr - min_value) / (max_value - min_value)
    arr = arr * 255

    return arr.clip(0, 255).astype(np.uint8)


def build_prompt(task: TargetTask) -> str:
    if task == TargetTask.CONSTRUCTION_DECADE:
        return "<image> What decade was the building constructed?"

    if task == TargetTask.CONSTRUCTION_YEAR:
        return "<image> What year was this building constructed?"

    raise ValueError(f"Unknown task: {task}")


def construction_decade_label(year: int) -> str:
    decade = year // 10 * 10
    return f"{decade}s"


def get_target(record: dict, task: TargetTask) -> str:
    attrs = record["buildings"][0]["attributes"]
    year = int(attrs["construction_year"])

    if task == TargetTask.CONSTRUCTION_DECADE:
        return construction_decade_label(year)

    if task == TargetTask.CONSTRUCTION_YEAR:
        return str(year)

    raise ValueError(f"Unknown task: {task}")
