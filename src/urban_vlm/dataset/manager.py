import logging
from pathlib import Path

import geopandas as gpd
import rasterio
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)

from urban_vlm.dataset.config import PrepareDataConfig
from urban_vlm.dataset.jsonl import write_jsonl_record
from urban_vlm.dataset.records import RasterInfo, build_jsonl_record
from urban_vlm.dataset.schema import BuildingField
from urban_vlm.dataset.splits import split_jsonl_by_group

logger = logging.getLogger(__name__)


def prepare_jsonl_dataset(cfg: PrepareDataConfig) -> int:
    input_file = cfg.inputs.matched_buildings_file

    buildings = gpd.read_parquet(input_file)
    buildings = buildings[buildings[BuildingField.TILE_PATH].notna()].copy()

    if buildings.crs is None:
        raise ValueError(f"Matched buildings file has no CRS: {input_file}")

    max_records = cfg.records.max_records
    if max_records is not None:
        buildings = buildings.head(max_records).copy()

    grouped = buildings.groupby(BuildingField.TILE_PATH, sort=False)
    total = len(buildings)
    count = 0

    logger.info(
        "Preparing JSONL from %s buildings across %s raster tiles.",
        len(buildings),
        grouped.ngroups,
    )

    progress = Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
    )

    output_file = cfg.outputs.all_jsonl
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with progress:
        task = progress.add_task("Writing JSONL", total=total)

        with output_file.open("w", encoding="utf-8") as f:
            for tile_path, tile_buildings in grouped:
                tile_path = Path(tile_path)

                with rasterio.open(tile_path) as src:
                    raster = RasterInfo(
                        path=str(tile_path),
                        transform=src.transform,
                        width=src.width,
                        height=src.height,
                        crs=str(src.crs),
                    )

                    # Important: if the matched buildings CRS differs from raster CRS,
                    # reproject this tile's buildings before computing pixel geometry.
                    tile_buildings = _ensure_buildings_match_raster_crs(
                        tile_buildings,
                        raster_crs=raster.crs,
                    )

                    for _, row in tile_buildings.iterrows():
                        record = build_jsonl_record(
                            row,
                            raster=raster,
                            source_crs=tile_buildings.crs.name,
                            crop_padding_ratio=cfg.crop.padding_ratio,
                        )

                        write_jsonl_record(f, record)

                        count += 1
                        progress.update(task, completed=count)

    logger.info(
        "Wrote %s JSONL records to %s.",
        count,
        output_file,
    )

    if cfg.split.enabled:
        split_jsonl_by_group(
            input_jsonl=cfg.outputs.all_jsonl,
            train_jsonl=cfg.outputs.train_jsonl,
            val_jsonl=cfg.outputs.val_jsonl,
            test_jsonl=cfg.outputs.test_jsonl,
            group_key=cfg.split.group_key,
            train_fraction=cfg.split.train,
            val_fraction=cfg.split.val,
            test_fraction=cfg.split.test,
            seed=cfg.split.seed,
        )

    return count


def _ensure_buildings_match_raster_crs(
    buildings: gpd.GeoDataFrame,
    *,
    raster_crs: str,
) -> gpd.GeoDataFrame:
    if buildings.crs is None:
        raise ValueError("Buildings GeoDataFrame has no CRS.")

    if str(buildings.crs) == raster_crs:
        return buildings

    return buildings.to_crs(raster_crs)
