import logging
from pathlib import Path

import geopandas as gpd
import rasterio
from shapely.geometry import box

from urban_vlm.dataset.schema import BuildingField

logger = logging.getLogger(__name__)


def build_tile_index(
    tile_dir: str | Path,
    *,
    file_glob: str = "*.tif",
) -> gpd.GeoDataFrame:
    """
    Build a spatial index of raster tile footprints.

    Each tile record contains:
    - tile_id
    - tile_path
    - geometry footprint
    """
    tile_dir = Path(tile_dir)
    tile_paths = sorted(tile_dir.glob(file_glob))

    if not tile_paths:
        raise FileNotFoundError(
            f"No tile files found in {tile_dir} matching {file_glob!r}."
        )

    records = []
    expected_crs = None

    for tile_path in tile_paths:
        with rasterio.open(tile_path) as src:
            if src.crs is None:
                raise ValueError(f"Tile has no CRS: {tile_path}")

            crs = src.crs.to_string()
            bounds = src.bounds

            if expected_crs is None:
                expected_crs = crs
            elif crs != expected_crs:
                raise ValueError(
                    "Found rasters with different CRS values. "
                    f"Expected {expected_crs}, got {crs} for {tile_path}"
                )

            records.append(
                {
                    str(BuildingField.TILE_ID): tile_path.stem,
                    str(BuildingField.TILE_PATH): str(tile_path),
                    "geometry": box(
                        bounds.left,
                        bounds.bottom,
                        bounds.right,
                        bounds.top,
                    ),
                }
            )

    tiles = gpd.GeoDataFrame(
        records,
        geometry="geometry",
        crs=expected_crs,
    )

    logger.info(
        "Built tile index with %s tiles from %s.",
        len(tiles),
        tile_dir,
    )

    return tiles
