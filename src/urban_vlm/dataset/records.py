import hashlib
from dataclasses import dataclass
from typing import Any

import pandas as pd
from affine import Affine

from urban_vlm.dataset.config import CropConfig
from urban_vlm.dataset.geometry import (
    building_geometry_for_crop,
    crop_spec_from_pixel_bbox,
    geometry_pixel_bbox,
    geometry_to_pixel_geometry,
    transform_point,
)
from urban_vlm.dataset.schema import BuildingField
from urban_vlm.eubucco.schema import EubuccoField


@dataclass(frozen=True)
class RasterInfo:
    path: str
    transform: Affine
    width: int
    height: int
    crs: str


@dataclass(frozen=True)
class JsonlRecordBuildResult:
    record: dict[str, Any] | None
    skip_reason: str | None = None
    is_clipped: bool = False


def build_jsonl_record(
    row: pd.Series,
    *,
    raster: RasterInfo,
    source_crs: str,
    crop: CropConfig,
) -> dict[str, Any] | None:
    return build_jsonl_record_result(
        row,
        raster=raster,
        source_crs=source_crs,
        crop=crop,
    ).record


def build_jsonl_record_result(
    row: pd.Series,
    *,
    raster: RasterInfo,
    source_crs: str,
    crop: CropConfig,
) -> JsonlRecordBuildResult:
    tile_id = _get_optional(row, str(BuildingField.TILE_ID))
    geometry = row.geometry

    tile_pixel_geometry = geometry_to_pixel_geometry(
        geometry,
        raster.transform,
    )
    tile_pixel_bbox = geometry_pixel_bbox(tile_pixel_geometry)

    crop_spec = crop_spec_from_pixel_bbox(
        tile_pixel_bbox,
        mode=crop.mode,
        padding_ratio=crop.padding_ratio,
        min_padding_px=crop.min_padding_px,
        max_padding_px=crop.max_padding_px,
        fixed_size_px=crop.fixed_size_px,
        adaptive_scale=crop.adaptive_scale,
        min_size_px=crop.min_size_px,
        max_size_px=crop.max_size_px,
        square=crop.square,
        drop_edge_crops=crop.drop_edge_crops,
        image_width=raster.width,
        image_height=raster.height,
    )

    if crop_spec is None:
        return JsonlRecordBuildResult(
            record=None,
            skip_reason=_infer_crop_skip_reason(
                tile_pixel_bbox,
                raster=raster,
                crop=crop,
            ),
        )

    building_id = str(row[str(EubuccoField.id)])
    record_id = _make_record_id(
        building_id=building_id,
        tile_id=str(tile_id) if tile_id is not None else None,
        crop_bounds=crop_spec.bounds,
    )

    building = build_building_record(
        row,
        source_crs=source_crs,
        transform=raster.transform,
        crop_bounds=crop_spec.bounds,
    )

    record: dict[str, Any] = {
        "id": record_id,
        "image": raster.path,
        "crop": {
            "type": "single",
            "mode": crop.mode,
            "bounds": crop_spec.bounds,
            "width": crop_spec.width,
            "height": crop_spec.height,
            "is_clipped": crop_spec.is_clipped,
        },
        "buildings": [
            building,
        ],
    }

    return JsonlRecordBuildResult(
        record=record,
        is_clipped=bool(crop_spec.is_clipped),
    )


def build_building_record(
    row: pd.Series,
    *,
    source_crs: str,
    transform: Affine,
    crop_bounds: list[int],
) -> dict[str, Any]:
    geometry = row.geometry
    centroid = geometry.centroid

    building_pixel_geometry = building_geometry_for_crop(
        geometry,
        transform,
        crop_bounds=crop_bounds,
    )

    construction_year = _get_optional(
        row,
        str(EubuccoField.construction_year),
    )

    return {
        "building_id": str(row[str(EubuccoField.id)]),
        "geometry": {
            "footprint": building_pixel_geometry.footprint,
            "bbox": building_pixel_geometry.bbox,
        },
        "location": {
            "crs": "EPSG:4326",
            "center": transform_point(
                centroid.x,
                centroid.y,
                source_crs=source_crs,
                target_crs="EPSG:4326",
            ),
        },
        "attributes": {
            "region_id": _get_optional(row, str(EubuccoField.region_id)),
            "city_id": _get_optional(row, str(EubuccoField.city_id)),
            "type": _get_optional(row, str(EubuccoField.type)),
            "subtype": _get_optional(row, str(EubuccoField.subtype)),
            "height": _get_optional(row, str(EubuccoField.height)),
            "floors": _get_optional(row, str(EubuccoField.floors)),
            "construction_year": construction_year,
            "construction_decade": _construction_decade(construction_year),
            "area_m2": float(geometry.area) if geometry is not None else None,
        },
    }


def _infer_crop_skip_reason(
    tile_pixel_bbox,
    *,
    raster: RasterInfo,
    crop: CropConfig,
) -> str:
    if crop.drop_edge_crops:
        unclipped_crop_spec = crop_spec_from_pixel_bbox(
            tile_pixel_bbox,
            mode=crop.mode,
            padding_ratio=crop.padding_ratio,
            min_padding_px=crop.min_padding_px,
            max_padding_px=crop.max_padding_px,
            fixed_size_px=crop.fixed_size_px,
            adaptive_scale=crop.adaptive_scale,
            min_size_px=crop.min_size_px,
            max_size_px=crop.max_size_px,
            square=crop.square,
            drop_edge_crops=False,
            image_width=raster.width,
            image_height=raster.height,
        )

        if unclipped_crop_spec is not None and unclipped_crop_spec.is_clipped:
            return "clipped_crop"

    return "invalid_crop"


def _make_record_id(
    *,
    building_id: str,
    tile_id: str | None,
    crop_bounds: list[int],
) -> str:
    key = "|".join(
        [
            building_id,
            tile_id or "",
            ",".join(str(value) for value in crop_bounds),
        ]
    )

    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]


def _construction_decade(
    year: int | float | str | None,
) -> int | None:
    if year is None or pd.isna(year):
        return None

    year_int = int(year)
    return year_int // 10 * 10


def _require_value(row: pd.Series, column: str) -> str:
    if column not in row:
        raise KeyError(f"Missing required column: {column!r}")

    value = row[column]

    if _is_missing(value):
        raise ValueError(f"Missing required value for column: {column!r}")

    return str(value)


def _get_optional(row: pd.Series, column: str):
    if column not in row:
        return None

    value = row[column]

    if _is_missing(value):
        return None

    return value


def _is_missing(value) -> bool:
    try:
        return bool(pd.isna(value))
    except ValueError:
        return False
