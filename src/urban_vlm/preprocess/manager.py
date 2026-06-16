from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import TypeVar

import geopandas as gpd
import pandas as pd
from rich.align import Align
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
)
from rich.table import Table

from urban_vlm.eubucco.io import EubuccoReadStats, read_eubucco_buildings
from urban_vlm.preprocess.config import EubuccoPreprocessConfig, PreprocessConfig
from urban_vlm.preprocess.match import match_buildings_to_tiles
from urban_vlm.preprocess.tiles import build_tile_index
from urban_vlm.utils import write_geodataframe

T = TypeVar("T")


@dataclass(frozen=True)
class EubuccoReadSummary:
    files: int
    raw_rows: int
    final_rows: int
    dropped_missing_or_empty_geometry: int
    repaired_invalid_geometry: int
    dropped_invalid_geometry: int
    dropped_outside_aoi: int
    dissolved_rows_removed: int
    dropped_missing_construction_year: int
    dropped_below_min_area: int
    dropped_above_max_area: int

    @property
    def total_dropped(self) -> int:
        return self.raw_rows - self.final_rows


@dataclass(frozen=True)
class PreprocessStepSummary:
    name: str
    rows: int | None
    output: Path | None
    elapsed_seconds: float


def preprocess_all(cfg: PreprocessConfig, *, show_progress: bool = True) -> None:
    console = Console()
    step_summaries: list[PreprocessStepSummary] = []

    tiles, elapsed = _run_status_step(
        console,
        "Build tile index",
        show_progress=show_progress,
        fn=lambda: build_tile_index(cfg.tiles),
    )
    step_summaries.append(
        PreprocessStepSummary(
            name="Build tile index",
            rows=len(tiles),
            output=None,
            elapsed_seconds=elapsed,
        )
    )

    elapsed = _run_write_step(
        console,
        "Write tile index",
        show_progress=show_progress,
        fn=lambda: write_geodataframe(tiles, cfg.outputs.tile_index_file),
    )
    step_summaries.append(
        PreprocessStepSummary(
            name="Write tile index",
            rows=len(tiles),
            output=Path(cfg.outputs.tile_index_file),
            elapsed_seconds=elapsed,
        )
    )

    if show_progress:
        console.print()
        console.rule("[dim]Read EUBUCCO buildings[/dim]", style="dim")

    start = perf_counter()
    buildings, eubucco_summary = _read_all_eubucco_buildings(
        cfg.eubucco,
        tiles,
        console=console,
        show_progress=show_progress,
    )
    elapsed = perf_counter() - start

    if show_progress:
        console.print(
            f"[dim]Finished reading EUBUCCO buildings: "
            f"{len(buildings):,} row(s) in {elapsed:.1f}s[/dim]"
        )

    step_summaries.append(
        PreprocessStepSummary(
            name="Read EUBUCCO buildings",
            rows=len(buildings),
            output=None,
            elapsed_seconds=elapsed,
        )
    )

    elapsed = _run_write_step(
        console,
        "Write cleaned buildings",
        show_progress=show_progress,
        fn=lambda: write_geodataframe(
            buildings,
            cfg.outputs.cleaned_buildings_file,
        ),
    )
    step_summaries.append(
        PreprocessStepSummary(
            name="Write cleaned buildings",
            rows=len(buildings),
            output=Path(cfg.outputs.cleaned_buildings_file),
            elapsed_seconds=elapsed,
        )
    )

    matched, elapsed = _run_status_step(
        console,
        f"Match buildings to tiles ({cfg.match.strategy})",
        show_progress=show_progress,
        fn=lambda: match_buildings_to_tiles(
            buildings=buildings,
            tiles=tiles,
            strategy=cfg.match.strategy,
            keep_unmatched=cfg.match.keep_unmatched,
            target_crs=cfg.match.target_crs,
        ),
    )
    step_summaries.append(
        PreprocessStepSummary(
            name="Match buildings to tiles",
            rows=len(matched),
            output=None,
            elapsed_seconds=elapsed,
        )
    )

    elapsed = _run_write_step(
        console,
        "Write matched buildings",
        show_progress=show_progress,
        fn=lambda: write_geodataframe(
            matched,
            cfg.outputs.matched_buildings_file,
        ),
    )
    step_summaries.append(
        PreprocessStepSummary(
            name="Write matched buildings",
            rows=len(matched),
            output=Path(cfg.outputs.matched_buildings_file),
            elapsed_seconds=elapsed,
        )
    )

    if show_progress:
        console.print()
        _print_summary(console, step_summaries, eubucco_summary)


def _run_status_step(
    console: Console,
    name: str,
    *,
    show_progress: bool,
    fn: Callable[[], T],
) -> tuple[T, float]:
    if show_progress:
        console.print()
        console.rule(f"[dim]{name}[/dim]", style="dim")

    start = perf_counter()

    if show_progress:
        with console.status(f"[dim]{name}[/dim]", spinner="dots"):
            result = fn()
    else:
        result = fn()

    elapsed = perf_counter() - start

    if show_progress:
        console.print(f"[dim]Finished {name} in {elapsed:.1f}s[/dim]")

    return result, elapsed


def _run_write_step(
    console: Console,
    name: str,
    *,
    show_progress: bool,
    fn: Callable[[], None],
) -> float:
    if show_progress:
        console.print()
        console.rule(f"[dim]{name}[/dim]", style="dim")

    start = perf_counter()

    if show_progress:
        with console.status(f"[dim]{name}[/dim]", spinner="dots"):
            fn()
    else:
        fn()

    elapsed = perf_counter() - start

    if show_progress:
        console.print(f"[dim]Finished {name} in {elapsed:.1f}s[/dim]")

    return elapsed


def _read_all_eubucco_buildings(
    cfg: EubuccoPreprocessConfig,
    tiles: gpd.GeoDataFrame,
    *,
    console: Console,
    show_progress: bool = True,
) -> tuple[gpd.GeoDataFrame, EubuccoReadSummary]:
    paths = sorted(cfg.input_dir.glob(cfg.file_glob))

    if not paths:
        raise ValueError("No EUBUCCO input files were provided.")

    if tiles.crs is None:
        raise ValueError("Tiles GeoDataFrame has no CRS.")

    frames: list[gpd.GeoDataFrame] = []
    stats: list[EubuccoReadStats] = []

    aoi = _union_geometries(tiles)
    tile_crs = str(tiles.crs)

    if show_progress:
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[dim]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            console=console,
            transient=True,
        )

        with progress:
            task_id = progress.add_task("Reading EUBUCCO files", total=len(paths))

            for path in paths:
                progress.update(task_id, description=f"Reading {path.name}")

                result = read_eubucco_buildings(
                    path,
                    aoi=aoi,
                    aoi_crs=tile_crs,
                    target_crs=tile_crs,
                    dissolve_by_id=cfg.dissolve_by_id,
                    require_construction_year=cfg.require_construction_year,
                    min_area_m2=cfg.min_area_m2,
                    max_area_m2=cfg.max_area_m2,
                )

                frames.append(result.buildings)
                stats.append(result.stats)
                progress.advance(task_id)
    else:
        for path in paths:
            result = read_eubucco_buildings(
                path,
                aoi=aoi,
                aoi_crs=tile_crs,
                target_crs=tile_crs,
                dissolve_by_id=cfg.dissolve_by_id,
                require_construction_year=cfg.require_construction_year,
                min_area_m2=cfg.min_area_m2,
                max_area_m2=cfg.max_area_m2,
            )

            frames.append(result.buildings)
            stats.append(result.stats)

    if len(frames) == 1:
        buildings = frames[0]
    else:
        buildings = gpd.GeoDataFrame(
            pd.concat(frames, ignore_index=True),
            geometry=frames[0].geometry.name,
            crs=frames[0].crs,
        )

    return buildings, _summarize_eubucco_reads(stats)


def _summarize_eubucco_reads(stats: Sequence[EubuccoReadStats]) -> EubuccoReadSummary:
    return EubuccoReadSummary(
        files=len(stats),
        raw_rows=sum(item.raw_rows for item in stats),
        final_rows=sum(item.final_rows for item in stats),
        dropped_missing_or_empty_geometry=sum(
            item.dropped_missing_or_empty_geometry for item in stats
        ),
        repaired_invalid_geometry=sum(item.repaired_invalid_geometry for item in stats),
        dropped_invalid_geometry=sum(item.dropped_invalid_geometry for item in stats),
        dropped_outside_aoi=sum(item.dropped_outside_aoi for item in stats),
        dissolved_rows_removed=sum(item.dissolved_rows_removed for item in stats),
        dropped_missing_construction_year=sum(
            item.dropped_missing_construction_year for item in stats
        ),
        dropped_below_min_area=sum(item.dropped_below_min_area for item in stats),
        dropped_above_max_area=sum(item.dropped_above_max_area for item in stats),
    )


def _union_geometries(gdf: gpd.GeoDataFrame):
    geometry = gdf.geometry

    if hasattr(geometry, "union_all"):
        return geometry.union_all()

    return geometry.unary_union


def _print_summary(
    console: Console,
    step_summaries: Sequence[PreprocessStepSummary],
    eubucco_summary: EubuccoReadSummary,
) -> None:
    table = Table(
        title="Preprocessing summary",
        show_header=True,
        header_style="bold",
    )

    table.add_column("Metric")
    table.add_column("Value", justify="right")

    total_seconds = sum(summary.elapsed_seconds for summary in step_summaries)

    for summary in step_summaries:
        value = "—" if summary.rows is None else f"{summary.rows:,}"
        table.add_row(summary.name, value)

    table.add_section()
    table.add_row("EUBUCCO files", f"{eubucco_summary.files:,}")
    table.add_row("EUBUCCO raw rows", f"{eubucco_summary.raw_rows:,}")
    table.add_row("EUBUCCO final rows", f"{eubucco_summary.final_rows:,}")
    table.add_row("EUBUCCO total dropped", f"{eubucco_summary.total_dropped:,}")
    table.add_row(
        "Dropped missing/empty geometry",
        f"{eubucco_summary.dropped_missing_or_empty_geometry:,}",
    )
    table.add_row(
        "Repaired invalid geometry",
        f"{eubucco_summary.repaired_invalid_geometry:,}",
    )
    table.add_row(
        "Dropped invalid geometry",
        f"{eubucco_summary.dropped_invalid_geometry:,}",
    )
    table.add_row(
        "Dropped outside AOI",
        f"{eubucco_summary.dropped_outside_aoi:,}",
    )
    table.add_row(
        "Rows removed by dissolve",
        f"{eubucco_summary.dissolved_rows_removed:,}",
    )
    table.add_row(
        "Dropped missing construction year",
        f"{eubucco_summary.dropped_missing_construction_year:,}",
    )
    table.add_row(
        "Dropped below min area",
        f"{eubucco_summary.dropped_below_min_area:,}",
    )
    table.add_row(
        "Dropped above max area",
        f"{eubucco_summary.dropped_above_max_area:,}",
    )

    table.add_section()
    table.add_row("Total time", f"{total_seconds:.1f}s")

    console.print(Align.center(table))
