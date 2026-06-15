from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import geopandas as gpd
import pandas as pd
from rich.align import Align
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn
from rich.table import Table

from urban_vlm.eubucco.io import read_eubucco_buildings
from urban_vlm.preprocess.config import EubuccoPreprocessConfig, PreprocessConfig
from urban_vlm.preprocess.match import match_buildings_to_tiles
from urban_vlm.preprocess.tiles import build_tile_index
from urban_vlm.utils import write_geodataframe


@dataclass(frozen=True)
class PreprocessStepSummary:
    name: str
    rows: int | None
    output: Path | None
    elapsed_seconds: float


def preprocess_all(cfg: PreprocessConfig, *, show_progress: bool = True) -> None:
    """
    Full preprocessing pipeline.

    Reads EUBUCCO buildings, cleans/filters/dissolves them, builds or reads
    Bayern imagery tile index, matches buildings to tiles, and writes the
    matched building file.
    """
    console = Console()
    summaries: list[PreprocessStepSummary] = []

    tiles, elapsed = _run_status_step(
        console,
        "Build tile index",
        show_progress=show_progress,
        fn=lambda: build_tile_index(cfg.tiles),
    )
    summaries.append(
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
    summaries.append(
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
    buildings = _read_all_eubucco_buildings(
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

    summaries.append(
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
    summaries.append(
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
    summaries.append(
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
    summaries.append(
        PreprocessStepSummary(
            name="Write matched buildings",
            rows=len(matched),
            output=Path(cfg.outputs.matched_buildings_file),
            elapsed_seconds=elapsed,
        )
    )

    if show_progress:
        console.print()
        _print_summary(console, summaries)


def _run_status_step[T](
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
    console: Console | None = None,
    show_progress: bool = True,
) -> gpd.GeoDataFrame:
    paths = sorted(cfg.input_dir.glob(cfg.file_glob))

    if not paths:
        raise ValueError("No EUBUCCO input files were provided.")

    frames: list[gpd.GeoDataFrame] = []

    aoi = tiles.union_all()
    tile_crs = tiles.crs.name

    if show_progress:
        progress = Progress(
            TextColumn("[dim]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            console=console,
            transient=True,
        )

        with progress:
            task_id = progress.add_task(
                "Reading EUBUCCO files",
                total=len(paths),
            )

            for path in paths:
                progress.update(
                    task_id,
                    description=f"[dim]Reading {path.name}[/dim]",
                )

                frames.append(
                    _read_eubucco_file(
                        path,
                        cfg=cfg,
                        aoi=aoi,
                        tile_crs=tile_crs,
                    )
                )

                progress.advance(task_id)
    else:
        for path in paths:
            frames.append(
                _read_eubucco_file(
                    path,
                    cfg=cfg,
                    aoi=aoi,
                    tile_crs=tile_crs,
                )
            )

    if len(frames) == 1:
        return frames[0]

    return gpd.GeoDataFrame(
        pd.concat(frames, ignore_index=True),
        geometry=frames[0].geometry.name,
        crs=frames[0].crs,
    )


def _read_eubucco_file(
    path: Path,
    *,
    cfg: EubuccoPreprocessConfig,
    aoi,
    tile_crs: str,
) -> gpd.GeoDataFrame:
    return read_eubucco_buildings(
        path,
        aoi=aoi,
        aoi_crs=tile_crs,
        target_crs=tile_crs,
        dissolve_by_id=cfg.dissolve_by_id,
        require_construction_year=cfg.require_construction_year,
        min_area_m2=cfg.min_area_m2,
        max_area_m2=cfg.max_area_m2,
    )


def _print_summary(
    console: Console,
    summaries: Sequence[PreprocessStepSummary],
) -> None:
    table = Table(
        title="Preprocessing summary",
        show_header=True,
        header_style="bold",
    )

    table.add_column("Step")
    table.add_column("Rows", justify="right")
    table.add_column("Output")
    table.add_column("Time", justify="right")

    total_seconds = 0.0

    for summary in summaries:
        rows = "—" if summary.rows is None else f"{summary.rows:,}"
        output = "—" if summary.output is None else str(summary.output)
        elapsed = f"{summary.elapsed_seconds:.1f}s"

        table.add_row(summary.name, rows, output, elapsed)
        total_seconds += summary.elapsed_seconds

    table.add_section()
    table.add_row("Total", "—", "—", f"{total_seconds:.1f}s")

    console.print(Align.center(table))
