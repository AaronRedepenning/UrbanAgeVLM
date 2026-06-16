from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from PIL import Image
from rasterio.windows import Window


def resolve_image_path(path: str | Path, image_root: str | Path | None = None) -> Path:
    path = Path(path)

    if path.is_absolute() or image_root is None:
        return path

    return Path(image_root) / path


def crop_image_from_record(
    record: dict[str, Any],
    *,
    image_root: str | Path | None = None,
) -> Image.Image:
    image_path = resolve_image_path(record["image"], image_root=image_root)

    crop = record.get("crop")
    if not crop:
        raise ValueError(f"Record has no crop info: {record.get('id')}")

    bounds = crop.get("bounds")
    if bounds is None or len(bounds) != 4:
        raise ValueError(
            f"Expected crop bounds [x0, y0, x1, y1] for record "
            f"{record.get('id')}. Got: {bounds}"
        )

    x0, y0, x1, y1 = [int(value) for value in bounds]
    width = x1 - x0
    height = y1 - y0

    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid crop bounds for record {record.get('id')}: {bounds}")

    with rasterio.open(image_path) as src:
        window = Window(
            col_off=x0,
            row_off=y0,
            width=width,
            height=height,
        )

        array = src.read(
            indexes=_read_indexes(src.count),
            window=window,
            boundless=False,
        )

    array = _ensure_three_channels(array)
    array = _to_uint8(array)
    array = np.moveaxis(array, 0, -1)

    return Image.fromarray(array, mode="RGB")


def _read_indexes(band_count: int) -> list[int]:
    if band_count <= 0:
        raise ValueError("Raster has no bands.")

    if band_count == 1:
        return [1]

    # Use RGB. If the raster has an alpha / NIR / extra band, ignore it.
    return [1, 2, 3]


def _ensure_three_channels(array: np.ndarray) -> np.ndarray:
    if array.ndim != 3:
        raise ValueError(
            f"Expected raster crop with shape (bands, height, width). "
            f"Got: {array.shape}"
        )

    if array.shape[0] == 1:
        return np.repeat(array, 3, axis=0)

    if array.shape[0] >= 3:
        return array[:3]

    raise ValueError(f"Expected at least one raster band. Got: {array.shape[0]}")


def _to_uint8(array: np.ndarray) -> np.ndarray:
    if array.dtype == np.uint8:
        return array

    if np.issubdtype(array.dtype, np.integer):
        return _integer_to_uint8(array)

    return _float_to_uint8(array)


def _integer_to_uint8(array: np.ndarray) -> np.ndarray:
    info = np.iinfo(array.dtype)

    if info.max <= 255 and info.min >= 0:
        return array.astype(np.uint8)

    array = array.astype(np.float32)
    array = (array - info.min) / (info.max - info.min)
    array = np.clip(array * 255.0, 0, 255)

    return array.astype(np.uint8)


def _float_to_uint8(array: np.ndarray) -> np.ndarray:
    array = array.astype(np.float32)

    finite = np.isfinite(array)
    if not finite.any():
        return np.zeros(array.shape, dtype=np.uint8)

    finite_values = array[finite]
    min_value = float(np.nanmin(finite_values))
    max_value = float(np.nanmax(finite_values))

    if max_value <= min_value:
        return np.zeros(array.shape, dtype=np.uint8)

    array = (array - min_value) / (max_value - min_value)
    array = np.clip(array * 255.0, 0, 255)

    return array.astype(np.uint8)
