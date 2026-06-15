from pathlib import Path

import typer

from urban_vlm.paligemma.train import train_paligemma


def main(
    config: Path = typer.Option(
        Path("configs/train.yaml"),
        "--config",
        "-c",
        help="Path to prepare config YAML.",
    ),
) -> None:
    train_paligemma()


if __name__ == "__main__":
    typer.run(main)
