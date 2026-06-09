import logging
from pathlib import Path

import geopandas as gpd
import pandas as pd

from urban_vlm.eubucco.io import read_eubucco_buildings
from urban_vlm.preprocess.match import match_buildings_to_tiles
from urban_vlm.preprocess.tiles import build_tile_index

logger = logging.getLogger(__name__)


def preprocess_buildings_and_match_tiles(
    cfg: dict,
) -> gpd.GeoDataFrame:
    """
    Full preprocessing pipeline.

    Reads EUBUCCO buildings, cleans/filters/dissolves them, builds or reads
    Bayern imagery tile index, matches buildings to tiles, and writes the
    matched building file.
    """
    logger.info("Starting preprocessing pipeline.")

    tiles = _get_tile_index(cfg)
    buildings = _read_all_eubucco_buildings(cfg, tiles)

    matched = match_buildings_to_tiles(
        buildings=buildings,
        tiles=tiles,
    )

    outputs_cfg = cfg["outputs"]
    matched_buildings_file = Path(outputs_cfg["matched_buildings_file"])
    _write_geodataframe(matched, matched_buildings_file)

    logger.info(
        "Wrote matched buildings to %s.",
        matched_buildings_file,
    )

    logger.info(
        "Preprocessing complete. Matched %s buildings.",
        len(matched),
    )

    return matched


def _read_all_eubucco_buildings(
    cfg: dict,
    tiles: gpd.GeoDataFrame,
    *,
    file_glob: str = "*.parquet",
) -> gpd.GeoDataFrame:
    frames: list[gpd.GeoDataFrame] = []

    eubucco_cfg = cfg["eubucco"]
    input_dir = Path(eubucco_cfg["input_dir"])
    dissolve_by_id = eubucco_cfg.get("dissolve_by_id", False)

    for path in input_dir.glob(file_glob):
        logger.info("Reading EUBUCCO buildings from %s.", path)

        buildings = read_eubucco_buildings(
            path,
            aoi=tiles.union_all(),
            aoi_crs=tiles.crs.name,
            target_crs=tiles.crs.name,
            dissolve_by_id=dissolve_by_id,
        )

        frames.append(buildings)

    if not frames:
        raise ValueError("No EUBUCCO input files were provided.")

    if len(frames) == 1:
        all_buildings = frames[0]
    else:
        all_buildings = gpd.GeoDataFrame(
            pd.concat(frames, ignore_index=True),
            geometry=frames[0].geometry.name,
            crs=frames[0].crs,
        )

    logger.info(
        "Read %s cleaned EUBUCCO buildings from %s files.",
        len(all_buildings),
        len(frames),
    )

    return all_buildings


def _get_tile_index(
    cfg: dict,
) -> gpd.GeoDataFrame:
    tile_cfg = cfg["tiles"]
    input_dir = Path(tile_cfg["input_dir"])

    logger.info("Building tile index from %s.", input_dir)

    return build_tile_index(
        input_dir,
    )


def _write_geodataframe(
    gdf: gpd.GeoDataFrame,
    path: str | Path,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    suffix = path.suffix.lower()

    if suffix in {".parquet", ".geoparquet"}:
        gdf.to_parquet(path)
        return

    if suffix in {".gpkg"}:
        gdf.to_file(path, driver="GPKG")
        return

    if suffix in {".geojson", ".json"}:
        gdf.to_file(path, driver="GeoJSON")
        return

    raise ValueError(f"Unsupported output file type: {path.suffix}")
