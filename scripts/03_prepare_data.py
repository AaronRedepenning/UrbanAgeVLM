from pathlib import Path

import typer

from urban_vlm.dataset import load_prepare_config, prepare_jsonl_dataset


def main(
    config: Path = typer.Option(
        Path("configs/prepare.yaml"),
        "--config",
        "-c",
        help="Path to prepare config YAML.",
    ),
) -> None:
    prepare_jsonl_dataset(load_prepare_config(config))


if __name__ == "__main__":
    typer.run(main)
