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

from urban_vlm.dataset.config import PrepareDataConfig
from urban_vlm.dataset.jsonl import write_jsonl_record
from urban_vlm.dataset.records import (
    RasterInfo,
    build_jsonl_record_result,
)
from urban_vlm.dataset.schema import BuildingField
from urban_vlm.dataset.splits import split_jsonl_by_group


@dataclass(frozen=True)
class PrepareJsonlSummary:
    input_file: Path
    output_file: Path
    input_rows: int
    rows_with_tiles: int
    rows_after_limit: int
    tile_count: int
    written: int
    skipped: int
    written_clipped: int
    skipped_clipped: int
    skip_reasons: dict[str, int]
    elapsed_seconds: float
    split_summary: dict[str, Any] | None = None


def prepare_all(
    cfg: PrepareDataConfig,
    *,
    show_progress: bool = True,
) -> PrepareJsonlSummary:
    console = Console()
    input_file = Path(cfg.inputs.matched_buildings_file)
    output_file = Path(cfg.outputs.all_jsonl)

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

    output_file.parent.mkdir(parents=True, exist_ok=True)

    (
        written,
        skipped,
        written_clipped,
        skipped_clipped,
        skip_reasons,
    ) = _write_jsonl_records(
        cfg,
        buildings=buildings,
        grouped=grouped,
        output_file=output_file,
        console=console,
        show_progress=show_progress,
    )

    split_summary: dict[str, Any] | None = None

    if cfg.split.enabled:
        split_summary = _run_split_step(
            cfg,
            console=console,
            show_progress=show_progress,
        )

    elapsed_total = perf_counter() - start_total

    summary = PrepareJsonlSummary(
        input_file=input_file,
        output_file=output_file,
        input_rows=input_rows,
        rows_with_tiles=rows_with_tiles,
        rows_after_limit=rows_after_limit,
        tile_count=tile_count,
        written=written,
        skipped=skipped,
        written_clipped=written_clipped,
        skipped_clipped=skipped_clipped,
        skip_reasons=dict(skip_reasons),
        elapsed_seconds=elapsed_total,
        split_summary=split_summary,
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

                        result = build_jsonl_record_result(
                            row,
                            raster=raster,
                            source_crs=str(tile_buildings.crs),
                            crop=cfg.crop,
                        )

                        if result.record is None:
                            skipped += 1

                            reason = result.skip_reason or "unknown"
                            skip_reasons[reason] += 1

                            if reason == "clipped_crop":
                                skipped_clipped += 1

                            if show_progress and task_id is not None:
                                progress.update(task_id, completed=seen)

                            continue

                        write_jsonl_record(f, result.record)

                        written += 1

                        if result.is_clipped:
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
    console: Console,
    show_progress: bool,
) -> dict[str, Any]:
    if show_progress:
        console.print()
        console.rule("[dim]Split JSONL dataset[/dim]", style="dim")

        with console.status("[dim]Writing train/val/test splits[/dim]", spinner="dots"):
            split_summary = split_jsonl_by_group(
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
    else:
        split_summary = split_jsonl_by_group(
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

    table.add_section()
    table.add_row("JSONL records written", f"{summary.written:,}")
    table.add_row("Skipped records", f"{summary.skipped:,}")
    table.add_row("Written clipped crops", f"{summary.written_clipped:,}")
    table.add_row("Skipped clipped crops", f"{summary.skipped_clipped:,}")

    invalid_crop_count = summary.skip_reasons.get("invalid_crop", 0)
    if invalid_crop_count:
        table.add_row("Skipped invalid crops", f"{invalid_crop_count:,}")

    unknown_skip_count = summary.skip_reasons.get("unknown", 0)
    if unknown_skip_count:
        table.add_row("Skipped unknown reason", f"{unknown_skip_count:,}")

    table.add_row("Output", str(summary.output_file))
    table.add_row("Time", f"{summary.elapsed_seconds:.1f}s")

    if summary.split_summary is not None:
        splits = summary.split_summary["splits"]

        table.add_section()
        table.add_row("Split group key", str(summary.split_summary["group_key"]))
        table.add_row("Split groups", f"{summary.split_summary['num_groups']:,}")

        for split_name in ("train", "val", "test"):
            split = splits[split_name]
            table.add_row(
                f"{split_name} split",
                f"{split['num_records']:,} records / {split['num_groups']:,} groups",
            )

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
