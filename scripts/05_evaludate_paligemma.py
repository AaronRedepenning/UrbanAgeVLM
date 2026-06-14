from pathlib import Path

import typer

from urban_vlm.paligemma.evaluate import evaluate_paligemma


def main(
    config: Path = typer.Option(
        Path("configs/evaluate.yaml"),
        "--config",
        "-c",
        help="Path to prepare config YAML.",
    ),
) -> None:
    evaluate_paligemma()


if __name__ == "__main__":
    typer.run(main)
