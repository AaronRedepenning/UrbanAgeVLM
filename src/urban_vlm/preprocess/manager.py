import logging

import geopandas as gpd
import pandas as pd

from urban_vlm.eubucco.io import read_eubucco_buildings
from urban_vlm.preprocess.config import EubuccoPreprocessConfig, PreprocessConfig
from urban_vlm.preprocess.match import match_buildings_to_tiles
from urban_vlm.preprocess.tiles import build_tile_index
from urban_vlm.utils import write_geodataframe

logger = logging.getLogger(__name__)


def preprocess_buildings_and_match_tiles(
    cfg: PreprocessConfig,
) -> gpd.GeoDataFrame:
    """
    Full preprocessing pipeline.

    Reads EUBUCCO buildings, cleans/filters/dissolves them, builds or reads
    Bayern imagery tile index, matches buildings to tiles, and writes the
    matched building file.
    """
    logger.info("Starting preprocessing pipeline.")

    tiles = build_tile_index(cfg.tiles)
    write_geodataframe(tiles, cfg.outputs.tile_index_file)

    buildings = _read_all_eubucco_buildings(cfg.eubucco, tiles)
    write_geodataframe(tiles, cfg.outputs.cleaned_buildings_file)

    matched = match_buildings_to_tiles(buildings=buildings, tiles=tiles)
    write_geodataframe(matched, cfg.outputs.matched_buildings_file)
    logger.info("Wrote matched buildings to %s.", cfg.outputs.matched_buildings_file)
    logger.info("Preprocessing complete. Matched %s buildings.", len(matched))

    return matched


def _read_all_eubucco_buildings(
    cfg: EubuccoPreprocessConfig,
    tiles: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    frames: list[gpd.GeoDataFrame] = []

    for path in cfg.input_dir.glob(cfg.file_glob):
        logger.info("Reading EUBUCCO buildings from %s.", path)

        buildings = read_eubucco_buildings(
            path,
            aoi=tiles.union_all(),
            aoi_crs=tiles.crs.name,
            target_crs=tiles.crs.name,
            dissolve_by_id=cfg.dissolve_by_id,
            require_construction_year=cfg.require_construction_year,
            min_area_m2=cfg.min_area_m2,
            max_area_m2=cfg.max_area_m2,
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
