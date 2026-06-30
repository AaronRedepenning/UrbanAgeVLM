from pathlib import Path

import typer

from urban_vlm.preprocess import load_preprocess_config, preprocess_all


def main(
    config: Path = typer.Option(
        Path("configs/preprocess/bayern.yaml"),
        "--config",
        "-c",
        exists=True,
        dir_okay=False,
        file_okay=True,
        readable=True,
        help="Path to preprocess config YAML.",
    ),
) -> None:
    preprocess_all(load_preprocess_config(config))


if __name__ == "__main__":
    typer.run(main)
