from pathlib import Path

import typer

from urban_vlm.paligemma import load_paligemma_config, predict_paligemma


def main(
    config: Path = typer.Option(
        Path("configs/predict/pg2-3b-pt-448.lora.yaml"),
        "--config",
        "-c",
        exists=True,
        dir_okay=False,
        file_okay=True,
        readable=True,
        help="Path to predict config YAML.",
    ),
) -> None:
    predict_paligemma(
        load_paligemma_config(config),
    )


if __name__ == "__main__":
    typer.run(main)
