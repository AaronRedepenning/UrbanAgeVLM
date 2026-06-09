import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import rasterio
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.patches import Rectangle
from rasterio.windows import Window

logger = logging.getLogger(__name__)


def save_crop_debug_image(
    record: dict[str, Any],
    output_path: str | Path,
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

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(image)

    for building in record["buildings"]:
        bbox = building["geometry"]["bbox"]
        bx_min, by_min, bx_max, by_max = bbox

        rect = Rectangle(
            (bx_min, by_min),
            bx_max - bx_min,
            by_max - by_min,
            fill=False,
            linewidth=2,
        )
        ax.add_patch(rect)

        for ring in building["geometry"]["footprint"]:
            polygon = MplPolygon(
                ring,
                closed=True,
                fill=False,
                linewidth=1.5,
            )
            ax.add_patch(polygon)

    title = record.get("id", "record")
    ax.set_title(title)
    ax.set_axis_off()

    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close(fig)


def _rasterio_to_display_image(image):
    """
    Convert rasterio array from [bands, height, width] to display image.
    """
    if image.ndim != 3:
        return image

    if image.shape[0] >= 3:
        image = image[:3]
        image = image.transpose(1, 2, 0)
        return image

    if image.shape[0] == 1:
        return image[0]

    return image.transpose(1, 2, 0)
