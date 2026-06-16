import math
from dataclasses import dataclass
from typing import Literal

from affine import Affine
from pyproj import Transformer
from shapely.affinity import translate
from shapely.geometry import MultiPolygon, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform as shapely_transform

PixelBounds = list[int]
PixelPoint = list[int]
PixelRing = list[PixelPoint]


@dataclass(frozen=True)
class CropSpec:
    bounds: PixelBounds
    width: int
    height: int
    is_clipped: bool = False


@dataclass(frozen=True)
class CropPlan:
    bounds: PixelBounds
    is_clipped: bool


@dataclass(frozen=True)
class BuildingPixelGeometry:
    footprint: list[PixelRing]
    bbox: PixelBounds


def transform_point(
    x: float,
    y: float,
    *,
    source_crs: str,
    target_crs: str = "EPSG:4326",
) -> list[float]:
    transformer = Transformer.from_crs(source_crs, target_crs, always_xy=True)
    lon, lat = transformer.transform(x, y)
    return [float(lon), float(lat)]


def geometry_to_pixel_geometry(
    geometry: BaseGeometry,
    transform: Affine,
) -> BaseGeometry:
    """
    Convert geometry from world/map coordinates to tile pixel coordinates.

    Pixel coordinates use image convention:
    - x = column
    - y = row
    - origin = top-left of the raster tile.
    """
    inverse_transform = ~transform

    def world_to_pixel(x: float, y: float, z: float | None = None):
        px, py = inverse_transform * (x, y)
        return px, py

    return shapely_transform(world_to_pixel, geometry)


def geometry_pixel_bbox(
    pixel_geometry: BaseGeometry,
) -> PixelBounds:
    minx, miny, maxx, maxy = pixel_geometry.bounds

    return [
        math.floor(minx),
        math.floor(miny),
        math.ceil(maxx),
        math.ceil(maxy),
    ]


def padding_from_bbox(
    bbox: PixelBounds,
    *,
    padding_ratio: float,
    min_padding_px: int = 0,
    max_padding_px: int | None = None,
) -> int:
    x_min, y_min, x_max, y_max = bbox

    width = max(0, x_max - x_min)
    height = max(0, y_max - y_min)

    padding = int(round(max(width, height) * padding_ratio))
    padding = max(padding, min_padding_px)

    if max_padding_px is not None:
        padding = min(padding, max_padding_px)

    return padding


def padded_bounds(
    bounds: PixelBounds,
    *,
    padding_px: int,
    image_width: int,
    image_height: int,
) -> PixelBounds:
    x_min, y_min, x_max, y_max = bounds

    return [
        max(0, x_min - padding_px),
        max(0, y_min - padding_px),
        min(image_width, x_max + padding_px),
        min(image_height, y_max + padding_px),
    ]


def crop_spec_from_pixel_bbox(
    bbox: PixelBounds,
    *,
    mode: Literal["percent", "fixed", "adaptive"],
    image_width: int,
    image_height: int,
    square: bool = True,
    drop_edge_crops: bool = False,
    padding_ratio: float = 0.5,
    min_padding_px: int = 0,
    max_padding_px: int | None = None,
    fixed_size_px: int | None = None,
    adaptive_scale: float = 2.0,
    min_size_px: int | None = None,
    max_size_px: int | None = None,
) -> CropSpec | None:
    """
    Build a crop around a building bbox.

    Returns None when the crop would extend outside the raster and
    drop_edge_crops=True.
    """

    if mode == "percent":
        raw_bounds = _percent_crop_bounds(
            bbox,
            padding_ratio=padding_ratio,
            min_padding_px=min_padding_px,
            max_padding_px=max_padding_px,
            square=square,
        )

    elif mode == "fixed":
        if fixed_size_px is None:
            raise ValueError("fixed_size_px is required when mode='fixed'.")

        center_x, center_y = bbox_center(bbox)
        raw_bounds = bounds_from_center(
            center_x,
            center_y,
            width=fixed_size_px,
            height=fixed_size_px if square else fixed_size_px,
        )

    elif mode == "adaptive":
        raw_bounds = _adaptive_crop_bounds(
            bbox,
            adaptive_scale=adaptive_scale,
            min_size_px=min_size_px,
            max_size_px=max_size_px,
            square=square,
        )

    else:
        raise ValueError(f"Unsupported crop mode: {mode!r}")

    is_inside = bounds_are_inside_image(
        raw_bounds,
        image_width=image_width,
        image_height=image_height,
    )

    if drop_edge_crops and not is_inside:
        return None

    crop_bounds = clamp_bounds_to_image(
        raw_bounds,
        image_width=image_width,
        image_height=image_height,
    )

    x_min, y_min, x_max, y_max = crop_bounds

    if x_max <= x_min or y_max <= y_min:
        return None

    return CropSpec(
        bounds=crop_bounds,
        width=x_max - x_min,
        height=y_max - y_min,
        is_clipped=not is_inside,
    )


def pixel_geometry_to_crop_coordinates(
    pixel_geometry: BaseGeometry,
    crop_bounds: PixelBounds,
) -> BaseGeometry:
    crop_x_min, crop_y_min, _, _ = crop_bounds

    return translate(
        pixel_geometry,
        xoff=-crop_x_min,
        yoff=-crop_y_min,
    )


def footprint_to_pixel_coordinates(
    pixel_geometry: BaseGeometry,
) -> list[PixelRing]:
    """
    Convert pixel-space Polygon/MultiPolygon to coordinate rings.

    Coordinates are assumed to already be in the desired pixel space,
    e.g. crop-relative pixels.
    """
    if isinstance(pixel_geometry, Polygon):
        return [_polygon_exterior_to_points(pixel_geometry)]

    if isinstance(pixel_geometry, MultiPolygon):
        return [_polygon_exterior_to_points(poly) for poly in pixel_geometry.geoms]

    raise TypeError(
        f"Expected Polygon or MultiPolygon geometry, got {pixel_geometry.geom_type!r}."
    )


def building_geometry_for_crop(
    geometry: BaseGeometry,
    transform: Affine,
    *,
    crop_bounds: PixelBounds,
) -> BuildingPixelGeometry:
    """
    Return building footprint and bbox in crop-relative pixel coordinates.
    """
    tile_pixel_geometry = geometry_to_pixel_geometry(geometry, transform)
    crop_pixel_geometry = pixel_geometry_to_crop_coordinates(
        tile_pixel_geometry,
        crop_bounds,
    )

    return BuildingPixelGeometry(
        footprint=footprint_to_pixel_coordinates(crop_pixel_geometry),
        bbox=geometry_pixel_bbox(crop_pixel_geometry),
    )


def _polygon_exterior_to_points(polygon: Polygon) -> PixelRing:
    return [[int(x), int(y)] for x, y in polygon.exterior.coords]


def bbox_width(bounds: PixelBounds) -> int:
    x_min, _, x_max, _ = bounds
    return max(0, x_max - x_min)


def bbox_height(bounds: PixelBounds) -> int:
    _, y_min, _, y_max = bounds
    return max(0, y_max - y_min)


def bbox_center(bounds: PixelBounds) -> tuple[float, float]:
    x_min, y_min, x_max, y_max = bounds
    return (x_min + x_max) / 2.0, (y_min + y_max) / 2.0


def clamp_bounds_to_image(
    bounds: PixelBounds,
    *,
    image_width: int,
    image_height: int,
) -> PixelBounds:
    x_min, y_min, x_max, y_max = bounds

    return [
        max(0, x_min),
        max(0, y_min),
        min(image_width, x_max),
        min(image_height, y_max),
    ]


def bounds_are_inside_image(
    bounds: PixelBounds,
    *,
    image_width: int,
    image_height: int,
) -> bool:
    x_min, y_min, x_max, y_max = bounds

    return x_min >= 0 and y_min >= 0 and x_max <= image_width and y_max <= image_height


def bounds_from_center(
    center_x: float,
    center_y: float,
    *,
    width: int,
    height: int,
) -> PixelBounds:
    x_min = int(round(center_x - width / 2))
    y_min = int(round(center_y - height / 2))
    x_max = x_min + width
    y_max = y_min + height

    return [x_min, y_min, x_max, y_max]


def square_bounds_around_center(
    bounds: PixelBounds,
) -> PixelBounds:
    center_x, center_y = bbox_center(bounds)
    size = max(bbox_width(bounds), bbox_height(bounds))

    return bounds_from_center(
        center_x,
        center_y,
        width=size,
        height=size,
    )


def _percent_crop_bounds(
    bbox: PixelBounds,
    *,
    padding_ratio: float,
    min_padding_px: int,
    max_padding_px: int | None,
    square: bool,
) -> PixelBounds:
    padding_px = padding_from_bbox(
        bbox,
        padding_ratio=padding_ratio,
        min_padding_px=min_padding_px,
        max_padding_px=max_padding_px,
    )

    x_min, y_min, x_max, y_max = bbox

    bounds = [
        x_min - padding_px,
        y_min - padding_px,
        x_max + padding_px,
        y_max + padding_px,
    ]

    if square:
        bounds = square_bounds_around_center(bounds)

    return bounds


def _adaptive_crop_bounds(
    bbox: PixelBounds,
    *,
    adaptive_scale: float,
    min_size_px: int | None,
    max_size_px: int | None,
    square: bool,
) -> PixelBounds:
    center_x, center_y = bbox_center(bbox)

    building_width = bbox_width(bbox)
    building_height = bbox_height(bbox)

    if square:
        size = int(round(max(building_width, building_height) * adaptive_scale))

        if min_size_px is not None:
            size = max(size, min_size_px)

        if max_size_px is not None:
            size = min(size, max_size_px)

        return bounds_from_center(
            center_x,
            center_y,
            width=size,
            height=size,
        )

    width = int(round(building_width * adaptive_scale))
    height = int(round(building_height * adaptive_scale))

    if min_size_px is not None:
        width = max(width, min_size_px)
        height = max(height, min_size_px)

    if max_size_px is not None:
        width = min(width, max_size_px)
        height = min(height, max_size_px)

    return bounds_from_center(
        center_x,
        center_y,
        width=width,
        height=height,
    )
