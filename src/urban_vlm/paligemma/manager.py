from pathlib import Path
from time import perf_counter
from typing import Any

from rich.align import Align
from rich.console import Console
from rich.table import Table

from urban_vlm.paligemma.config import PaliGemmaConfig
from urban_vlm.paligemma.inference import predict_jsonl


def predict_paligemma(
    cfg: PaliGemmaConfig,
    *,
    input_jsonl: Path | None = None,
    output_csv: Path | None = None,
    batch_size: int | None = None,
    show_progress: bool = True,
) -> list[dict[str, Any]]:
    console = Console()

    if show_progress:
        console.print()
        console.rule(f"[dim]Run PaliGemma prediction ({cfg.task})[/dim]", style="dim")

    start = perf_counter()

    predictions = predict_jsonl(
        cfg,
        input_jsonl=input_jsonl,
        output_csv=output_csv,
        batch_size=batch_size,
        show_progress=show_progress,
    )

    elapsed = perf_counter() - start

    if show_progress:
        _print_prediction_summary(
            console,
            cfg=cfg,
            predictions=predictions,
            output_csv=output_csv,
            elapsed_seconds=elapsed,
        )

    return predictions


def _print_prediction_summary(
    console: Console,
    *,
    cfg: PaliGemmaConfig,
    predictions: list[dict[str, Any]],
    output_csv: Path | None,
    elapsed_seconds: float,
) -> None:
    table = Table(
        title="PaliGemma prediction summary",
        show_header=True,
        header_style="bold",
    )

    table.add_column("Metric")
    table.add_column("Value", justify="right")

    table.add_row("Task", str(cfg.task))
    table.add_row("Model", cfg.model.model_id)
    table.add_row("Predictions", f"{len(predictions):,}")
    table.add_row("Output", "—" if output_csv is None else str(output_csv))
    table.add_row("Time", f"{elapsed_seconds:.1f}s")

    console.print()
    console.print(Align.center(table))
