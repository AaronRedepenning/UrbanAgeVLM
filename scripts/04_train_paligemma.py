from pathlib import Path

import typer

from urban_vlm.paligemma import load_paligemma_config, train_paligemma


def main(
    config: Path = typer.Option(
        Path("configs/train.yaml"),
        "--config",
        "-c",
        exists=True,
        dir_okay=False,
        file_okay=True,
        readable=True,
        help="Path to train config YAML.",
    ),
) -> None:
    train_paligemma(
        load_paligemma_config(config),
    )


if __name__ == "__main__":
    typer.run(main)
