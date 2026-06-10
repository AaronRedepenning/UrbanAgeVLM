import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.patches import Rectangle
from rasterio.windows import Window

logger = logging.getLogger(__name__)


def save_crop_debug_image(
    record: dict[str, Any],
    output_path: str | Path,
    *,
    title: str | None = None,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    image_path = record["image"]
    crop = record["crop"]

    x_min, y_min, x_max, y_max = crop["bounds"]

    window = Window.from_slices(
        (y_min, y_max),
        (x_min, x_max),
    )

    with rasterio.open(image_path) as src:
        image = src.read(window=window)

    image = _rasterio_to_display_image(image)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(image)

    for building in record.get("buildings", []):
        _draw_building_overlay(ax, building)

    ax.set_title(title or _make_title(record), fontsize=9)
    ax.set_axis_off()

    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close(fig)


def _draw_building_overlay(ax, building: dict[str, Any]) -> None:
    geometry = building.get("geometry", {})

    bbox = geometry.get("bbox")
    if bbox and len(bbox) == 4:
        x_min, y_min, x_max, y_max = bbox

        rect = Rectangle(
            (x_min, y_min),
            x_max - x_min,
            y_max - y_min,
            fill=False,
            linewidth=2,
        )
        ax.add_patch(rect)

    footprint = geometry.get("footprint", [])
    for ring in footprint:
        if not ring:
            continue

        polygon = MplPolygon(
            ring,
            closed=True,
            fill=False,
            linewidth=1.5,
        )
        ax.add_patch(polygon)


def _make_title(record: dict[str, Any]) -> str:
    record_id = record.get("id", "unknown")
    building = record.get("buildings", [{}])[0]
    attrs = building.get("attributes", {})

    year = attrs.get("construction_year")
    decade = attrs.get("construction_decade_label") or attrs.get("construction_decade")
    crop = record.get("crop", {})

    parts = [
        str(record_id),
        f"year={year}",
        f"decade={decade}",
        f"crop={crop.get('width')}x{crop.get('height')}",
    ]

    if crop.get("clipped"):
        parts.append(f"clipped={','.join(crop.get('clip_sides', []))}")

    return " | ".join(part for part in parts if part is not None)


def _rasterio_to_display_image(image: np.ndarray) -> np.ndarray:
    """
    Convert rasterio image from [bands, height, width] to something imshow can show.
    Handles 1-band and RGB-like rasters.
    """
    if image.ndim != 3:
        return image

    if image.shape[0] >= 3:
        image = image[:3].transpose(1, 2, 0)
        return _normalize_if_needed(image)

    if image.shape[0] == 1:
        return _normalize_if_needed(image[0])

    return _normalize_if_needed(image.transpose(1, 2, 0))


def _normalize_if_needed(image: np.ndarray) -> np.ndarray:
    """
    Matplotlib expects uint8 in [0, 255] or float in [0, 1].
    This keeps uint8 as-is and scales other numeric arrays for display.
    """
    if image.dtype == np.uint8:
        return image

    image = image.astype("float32")

    finite = np.isfinite(image)
    if not finite.any():
        return image

    min_value = float(np.nanmin(image))
    max_value = float(np.nanmax(image))

    if max_value <= min_value:
        return image

    return (image - min_value) / (max_value - min_value)
