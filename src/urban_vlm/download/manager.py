from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

from rich.align import Align
from rich.console import Console
from rich.table import Table

from urban_vlm.download.bayern import download_bayern
from urban_vlm.download.berlin import download_berlin
from urban_vlm.download.config import DownloadConfig
from urban_vlm.download.eubucco import download_eubucco
from urban_vlm.download.nuts import download_nuts


@dataclass(frozen=True)
class DownloadStepSummary:
    name: str
    enabled: bool
    file_count: int | None
    elapsed_seconds: float | None


def _run_step(
    console: Console,
    *,
    name: str,
    enabled: bool,
    show_progress: bool,
    download: Callable[[], Sequence[Path] | None],
) -> DownloadStepSummary:
    if not enabled:
        if show_progress:
            console.print(f"[dim]Skipping {name}[/dim]")

        return DownloadStepSummary(
            name=name,
            enabled=False,
            file_count=None,
            elapsed_seconds=None,
        )

    if show_progress:
        _print_step_start(console, name)

    start = perf_counter()
    paths = download()
    elapsed = perf_counter() - start

    file_count = len(paths) if paths is not None else None

    if show_progress:
        _print_step_done(
            console,
            name,
            file_count=file_count,
            elapsed_seconds=elapsed,
        )

    return DownloadStepSummary(
        name=name,
        enabled=True,
        file_count=file_count,
        elapsed_seconds=elapsed,
    )


def _print_step_start(console: Console, name: str) -> None:
    console.print()
    console.rule(f"[dim]{name}[/dim]", style="dim")


def _print_step_done(
    console: Console,
    name: str,
    *,
    file_count: int | None,
    elapsed_seconds: float,
) -> None:
    if file_count is None:
        console.print(f"[dim]Finished {name} in {elapsed_seconds:.1f}s[/dim]")
    else:
        console.print(
            f"[dim]Finished {name}: {file_count} file(s) in {elapsed_seconds:.1f}s[/dim]"
        )


def _print_summary(
    console: Console,
    summaries: Sequence[DownloadStepSummary],
) -> None:
    enabled_summaries = [summary for summary in summaries if summary.enabled]

    if not enabled_summaries:
        console.print("[dim]No downloads enabled[/dim]")
        return

    table = Table(
        title="Download summary",
        show_header=True,
        header_style="bold",
    )

    table.add_column("Step")
    table.add_column("Files", justify="right")
    table.add_column("Time", justify="right")

    total_files = 0
    total_seconds = 0.0

    for summary in enabled_summaries:
        files = "—" if summary.file_count is None else str(summary.file_count)
        seconds = (
            "—"
            if summary.elapsed_seconds is None
            else f"{summary.elapsed_seconds:.1f}s"
        )

        table.add_row(summary.name, files, seconds)

        if summary.file_count is not None:
            total_files += summary.file_count

        if summary.elapsed_seconds is not None:
            total_seconds += summary.elapsed_seconds

    table.add_section()
    table.add_row("Total", str(total_files), f"{total_seconds:.1f}s")

    console.print(Align.center(table))


def download_all(cfg: DownloadConfig) -> None:
    show_progress = cfg.download.show_progress
    console = Console()

    summaries: list[DownloadStepSummary] = []

    summaries.append(
        _run_step(
            console,
            name="NUTS",
            enabled=cfg.nuts.enabled,
            show_progress=show_progress,
            download=lambda: download_nuts(
                cfg.nuts,
                cfg.download,
            ),
        )
    )

    summaries.append(
        _run_step(
            console,
            name="EUBUCCO",
            enabled=cfg.eubucco.enabled,
            show_progress=show_progress,
            download=lambda: download_eubucco(
                cfg.eubucco,
                cfg.download,
            ),
        )
    )

    summaries.append(
        _run_step(
            console,
            name="Bayern",
            enabled=cfg.bayern.enabled,
            show_progress=show_progress,
            download=lambda: download_bayern(
                cfg.bayern,
                cfg.download,
            ),
        )
    )

    summaries.append(
        _run_step(
            console,
            name="Berlin",
            enabled=cfg.berlin.enabled,
            show_progress=show_progress,
            download=lambda: download_berlin(
                cfg.berlin,
                cfg.download,
            ),
        )
    )

    if show_progress:
        console.print()
        _print_summary(console, summaries)
