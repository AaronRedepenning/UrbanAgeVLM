import logging
from pathlib import Path

import geopandas as gpd
from urban_vlm.dataset.jsonl import write_jsonl

from urban_vlm.dataset.records import build_jsonl_record

logger = logging.getLogger(__name__)


def prepare_jsonl_dataset(cfg: dict) -> int:
    input_file = Path(cfg["input_file"])
    output_file = Path(cfg["output_file"])
    max_records = cfg.get("max_records", None)
    crop_cfg = cfg["crop"]
    crop_padding_ratio = crop_cfg["padding_ratio"]

    buildings = gpd.read_parquet(input_file)

    if buildings.crs is None:
        raise ValueError(f"Matched buildings file has no CRS: {input_file}")

    if max_records is not None:
        buildings = buildings.head(max_records).copy()

    source_crs = str(buildings.crs)

    records = (
        build_jsonl_record(
            row,
            source_crs=source_crs,
            crop_padding_ratio=crop_padding_ratio,
        )
        for _, row in buildings.iterrows()
    )

    count = write_jsonl(records, output_file)

    logger.info(
        "Wrote %s JSONL records to %s.",
        count,
        output_file,
    )

    return count
