from pathlib import Path
from typing import Any

from urban_vlm.paligemma.images import crop_image_from_record


def write_crop_image(
    record: dict[str, Any],
    *,
    output_dir: Path,
    image_format: str = "png",
    overwrite: bool = False,
    relative_to: Path | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    record_id = str(record["id"])
    suffix = _normalize_suffix(image_format)
    output_path = output_dir / f"{record_id}{suffix}"

    if overwrite or not output_path.exists():
        image = crop_image_from_record(record)
        image.save(output_path)

    updated_record = dict(record)
    updated_record["crop_image"] = _stored_path(
        output_path,
        relative_to=relative_to,
    )

    return updated_record


def _normalize_suffix(image_format: str) -> str:
    image_format = image_format.lower().lstrip(".")

    if image_format == "jpeg":
        return ".jpg"

    return f".{image_format}"


def _stored_path(path: Path, *, relative_to: Path | None) -> str:
    if relative_to is None:
        return str(path)

    try:
        return str(path.relative_to(relative_to))
    except ValueError:
        return str(path)
