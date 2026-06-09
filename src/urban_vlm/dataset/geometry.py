from dataclasses import dataclass

from affine import Affine
from shapely.affinity import translate
from shapely.geometry import MultiPolygon, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform as shapely_transform

PixelBounds = list[int]
PixelPoint = list[float]
PixelRing = list[PixelPoint]


@dataclass(frozen=True)
class CropSpec:
    bounds: PixelBounds
    width: int
    height: int


@dataclass(frozen=True)
class BuildingPixelGeometry:
    footprint: list[PixelRing]
    bbox: PixelBounds


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
        int(minx // 1),
        int(miny // 1),
        int(maxx + 1),
        int(maxy + 1),
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
    padding_ratio: float,
    min_padding_px: int,
    max_padding_px: int | None,
    image_width: int,
    image_height: int,
) -> CropSpec:
    padding_px = padding_from_bbox(
        bbox,
        padding_ratio=padding_ratio,
        min_padding_px=min_padding_px,
        max_padding_px=max_padding_px,
    )

    crop_bounds = padded_bounds(
        bbox,
        padding_px=padding_px,
        image_width=image_width,
        image_height=image_height,
    )

    x_min, y_min, x_max, y_max = crop_bounds

    return CropSpec(
        bounds=crop_bounds,
        width=x_max - x_min,
        height=y_max - y_min,
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
    return [[float(x), float(y)] for x, y in polygon.exterior.coords]
