from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import geopandas as gpd
import rasterio
from rich.align import Align
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)
from rich.table import Table

from urban_vlm.dataset.config import (
    CropConfig,
    CropImageOutputConfig,
    PrepareDataConfig,
    PrepareOutputsConfig,
)
from urban_vlm.dataset.crops import write_crop_image
from urban_vlm.dataset.jsonl import write_jsonl_record
from urban_vlm.dataset.records import (
    RasterInfo,
    build_jsonl_record_result,
)
from urban_vlm.dataset.schema import BuildingField
from urban_vlm.dataset.splits import split_jsonl_by_group


@dataclass(frozen=True)
class PrepareCropSummary:
    crop_name: str
    output_file: Path
    written: int
    skipped: int
    written_clipped: int
    skipped_clipped: int
    skip_reasons: dict[str, int]
    split_summary: dict[str, Any] | None = None


@dataclass(frozen=True)
class PrepareJsonlSummary:
    input_file: Path
    input_rows: int
    rows_with_tiles: int
    rows_after_limit: int
    tile_count: int
    crop_summaries: list[PrepareCropSummary]
    elapsed_seconds: float


def prepare_all(
    cfg: PrepareDataConfig,
    *,
    show_progress: bool = True,
) -> PrepareJsonlSummary:
    console = Console()
    input_file = Path(cfg.inputs.matched_buildings_file)

    if show_progress:
        console.print()
        console.rule("[dim]Prepare JSONL dataset[/dim]", style="dim")

    start_total = perf_counter()

    buildings, input_rows, rows_with_tiles, rows_after_limit = _read_input_buildings(
        cfg,
        console=console,
        show_progress=show_progress,
    )

    grouped = buildings.groupby(str(BuildingField.TILE_PATH), sort=False)
    tile_count = grouped.ngroups

    if show_progress:
        console.print(
            "[dim]"
            f"Preparing {rows_after_limit:,} building(s) "
            f"across {tile_count:,} raster tile(s)"
            "[/dim]"
        )

    crop_summaries: list[PrepareCropSummary] = []

    for crop in cfg.crops:
        outputs = cfg.get_crop_outputs(crop)
        image_config = cfg.get_crop_image_config(crop)

        outputs.all_jsonl.parent.mkdir(parents=True, exist_ok=True)
        if image_config.output_dir is not None:
            image_config.output_dir.mkdir(parents=True, exist_ok=True)

        (
            written,
            skipped,
            written_clipped,
            skipped_clipped,
            skip_reasons,
        ) = _write_jsonl_records(
            cfg,
            crop=crop,
            crop_image_config=image_config,
            buildings=buildings,
            grouped=grouped,
            output_file=outputs.all_jsonl,
            console=console,
            show_progress=show_progress,
        )

        split_summary: dict[str, Any] | None = None

        if cfg.split.enabled:
            split_summary = _run_split_step(
                cfg,
                outputs=outputs,
                console=console,
                show_progress=show_progress,
            )

        crop_summaries.append(
            PrepareCropSummary(
                crop_name=crop.name or "single",
                output_file=outputs.all_jsonl,
                written=written,
                skipped=skipped,
                written_clipped=written_clipped,
                skipped_clipped=skipped_clipped,
                skip_reasons=dict(skip_reasons),
                split_summary=split_summary,
            )
        )

    elapsed_total = perf_counter() - start_total

    summary = PrepareJsonlSummary(
        input_file=input_file,
        input_rows=input_rows,
        rows_with_tiles=rows_with_tiles,
        rows_after_limit=rows_after_limit,
        tile_count=tile_count,
        crop_summaries=crop_summaries,
        elapsed_seconds=elapsed_total,
    )

    if show_progress:
        console.print()
        _print_summary(console, summary)

    return summary


def _read_input_buildings(
    cfg: PrepareDataConfig,
    *,
    console: Console,
    show_progress: bool,
) -> tuple[gpd.GeoDataFrame, int, int, int]:
    input_file = Path(cfg.inputs.matched_buildings_file)

    if show_progress:
        with console.status(
            f"[dim]Reading matched buildings from {input_file}[/dim]",
            spinner="dots",
        ):
            buildings = gpd.read_parquet(input_file)
    else:
        buildings = gpd.read_parquet(input_file)

    input_rows = len(buildings)

    if buildings.crs is None:
        raise ValueError(f"Matched buildings file has no CRS: {input_file}")

    tile_path_col = str(BuildingField.TILE_PATH)

    buildings = buildings[buildings[tile_path_col].notna()].copy()
    rows_with_tiles = len(buildings)

    max_records = cfg.records.max_records
    if max_records is not None:
        buildings = buildings.head(max_records).copy()

    rows_after_limit = len(buildings)

    return buildings, input_rows, rows_with_tiles, rows_after_limit


def _write_jsonl_records(
    cfg: PrepareDataConfig,
    *,
    crop: CropConfig,
    crop_image_config: CropImageOutputConfig,
    buildings: gpd.GeoDataFrame,
    grouped,
    output_file: Path,
    console: Console,
    show_progress: bool,
) -> tuple[int, int, int, int, Counter[str]]:
    total = len(buildings)
    seen = 0
    written = 0
    skipped = 0
    written_clipped = 0
    skipped_clipped = 0
    skip_reasons: Counter[str] = Counter()

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[dim]{task.description}", justify="left"),
        BarColumn(),
        MofNCompleteColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=True,
    )

    if show_progress:
        context = progress
    else:
        context = _NullProgressContext()

    with context:
        task_id = None

        if show_progress:
            task_id = progress.add_task("Writing JSONL", total=total)

        with output_file.open("w", encoding="utf-8") as f:
            for tile_path, tile_buildings in grouped:
                tile_path = Path(tile_path)

                if show_progress and task_id is not None:
                    progress.update(
                        task_id,
                        description=f"Processing {tile_path.name}",
                    )

                with rasterio.open(tile_path) as src:
                    raster = RasterInfo(
                        path=str(tile_path),
                        transform=src.transform,
                        width=src.width,
                        height=src.height,
                        crs=str(src.crs),
                    )

                    tile_buildings = _ensure_buildings_match_raster_crs(
                        tile_buildings,
                        raster_crs=raster.crs,
                    )

                    for _, row in tile_buildings.iterrows():
                        seen += 1

                        row_records: list[tuple[CropConfig, dict[str, Any], bool]] = []
                        skip_reason: str | None = None
                        any_clipped_failure = False
                        all_crops_successful = True

                        for crop_option in cfg.crops:
                            result = build_jsonl_record_result(
                                row,
                                raster=raster,
                                source_crs=str(tile_buildings.crs),
                                crop=crop_option,
                            )

                            if result.record is None:
                                all_crops_successful = False
                                skip_reason = result.skip_reason or "unknown"
                                if skip_reason == "clipped_crop":
                                    any_clipped_failure = True
                                break

                            row_records.append(
                                (crop_option, result.record, result.is_clipped)
                            )

                        if not all_crops_successful:
                            skipped += 1
                            reason = skip_reason or "unknown"
                            skip_reasons[reason] += 1

                            if any_clipped_failure:
                                skipped_clipped += 1

                            if show_progress and task_id is not None:
                                progress.update(task_id, completed=seen)

                            continue

                        current_record = next(
                            (
                                record,
                                is_clipped,
                            )
                            for crop_option, record, is_clipped in row_records
                            if crop_option.name == crop.name
                        )

                        record, is_clipped = current_record

                        if crop_image_config.enabled:
                            record = write_crop_image(
                                record,
                                output_dir=crop_image_config.output_dir,
                                image_format=crop_image_config.image_format,
                                overwrite=crop_image_config.overwrite,
                                relative_to=crop_image_config.relative_to,
                            )

                        write_jsonl_record(f, record)

                        written += 1

                        if is_clipped:
                            written_clipped += 1

                        if show_progress and task_id is not None:
                            progress.update(task_id, completed=seen)

    if show_progress:
        console.print(
            "[dim]"
            f"Finished JSONL: {written:,} written, "
            f"{skipped:,} skipped, "
            f"{written_clipped:,} written clipped, "
            f"{skipped_clipped:,} skipped clipped"
            "[/dim]"
        )

    return written, skipped, written_clipped, skipped_clipped, skip_reasons


def _run_split_step(
    cfg: PrepareDataConfig,
    *,
    outputs: PrepareOutputsConfig,
    console: Console,
    show_progress: bool,
) -> dict[str, Any]:
    if show_progress:
        console.print()
        console.rule("[dim]Split JSONL dataset[/dim]", style="dim")

        with console.status("[dim]Writing train/val/test splits[/dim]", spinner="dots"):
            split_summary = split_jsonl_by_group(
                input_jsonl=outputs.all_jsonl,
                train_jsonl=outputs.train_jsonl,
                val_jsonl=outputs.val_jsonl,
                test_jsonl=outputs.test_jsonl,
                group_key=cfg.split.group_key,
                train_fraction=cfg.split.train,
                val_fraction=cfg.split.val,
                test_fraction=cfg.split.test,
                seed=cfg.split.seed,
            )
    else:
        split_summary = split_jsonl_by_group(
            input_jsonl=outputs.all_jsonl,
            train_jsonl=outputs.train_jsonl,
            val_jsonl=outputs.val_jsonl,
            test_jsonl=outputs.test_jsonl,
            group_key=cfg.split.group_key,
            train_fraction=cfg.split.train,
            val_fraction=cfg.split.val,
            test_fraction=cfg.split.test,
            seed=cfg.split.seed,
        )

    if show_progress:
        splits = split_summary["splits"]
        console.print(
            "[dim]"
            f"Finished splits: "
            f"train={splits['train']['num_records']:,}, "
            f"val={splits['val']['num_records']:,}, "
            f"test={splits['test']['num_records']:,}"
            "[/dim]"
        )

    return split_summary


def _print_summary(console: Console, summary: PrepareJsonlSummary) -> None:
    table = Table(
        title="JSONL preparation summary",
        show_header=True,
        header_style="bold",
    )

    table.add_column("Metric")
    table.add_column("Value", justify="right")

    table.add_row("Input rows", f"{summary.input_rows:,}")
    table.add_row("Rows with tiles", f"{summary.rows_with_tiles:,}")
    table.add_row("Rows after limit", f"{summary.rows_after_limit:,}")
    table.add_row("Raster tiles", f"{summary.tile_count:,}")

    for crop_summary in summary.crop_summaries:
        table.add_section()
        table.add_row("Crop", crop_summary.crop_name)
        table.add_row("JSONL records written", f"{crop_summary.written:,}")
        table.add_row("Skipped records", f"{crop_summary.skipped:,}")
        table.add_row("Written clipped crops", f"{crop_summary.written_clipped:,}")
        table.add_row("Skipped clipped crops", f"{crop_summary.skipped_clipped:,}")

        invalid_crop_count = crop_summary.skip_reasons.get("invalid_crop", 0)
        if invalid_crop_count:
            table.add_row("Skipped invalid crops", f"{invalid_crop_count:,}")

        unknown_skip_count = crop_summary.skip_reasons.get("unknown", 0)
        if unknown_skip_count:
            table.add_row("Skipped unknown reason", f"{unknown_skip_count:,}")

        table.add_row("Output", str(crop_summary.output_file))

        if crop_summary.split_summary is not None:
            splits = crop_summary.split_summary["splits"]

            table.add_section()
            table.add_row(
                "Split group key", str(crop_summary.split_summary["group_key"])
            )
            table.add_row(
                "Split groups", f"{crop_summary.split_summary['num_groups']:,}"
            )

            for split_name in ("train", "val", "test"):
                split = splits[split_name]
                table.add_row(
                    f"{split_name} split",
                    f"{split['num_records']:,} records / {split['num_groups']:,} groups",
                )

    table.add_section()
    table.add_row("Time", f"{summary.elapsed_seconds:.1f}s")

    console.print(Align.center(table))


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


class _NullProgressContext:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False
