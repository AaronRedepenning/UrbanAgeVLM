from pathlib import Path

import typer

from urban_vlm.download import download_all, load_download_config


def main(
    config: Path = typer.Option(
        Path("configs/download.yaml"),
        "--config",
        "-c",
        exists=True,
        dir_okay=False,
        file_okay=True,
        readable=True,
        help="Path to download config YAML.",
    ),
) -> None:
    download_all(load_download_config(config))


if __name__ == "__main__":
    typer.run(main)
