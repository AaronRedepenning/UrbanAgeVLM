from pathlib import Path

import typer

from urban_vlm.preprocess import load_preprocess_config, preprocess_all


def main(
    config: Path = typer.Option(
        Path("configs/preprocess.yaml"),
        "--config",
        "-c",
        help="Path to preprocess config YAML.",
    ),
) -> None:
    preprocess_all(load_preprocess_config(config))


if __name__ == "__main__":
    typer.run(main)
