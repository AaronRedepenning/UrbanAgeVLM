from pathlib import Path
from typing import Any

import numpy as np
import typer

from urban_vlm.dataset.jsonl import read_jsonl, write_jsonl
from urban_vlm.eubucco.schema import EubuccoField
from urban_vlm.paligemma import PaliGemmaConfig, load_paligemma_config
from urban_vlm.paligemma.config import PaliGemmaTask
from urban_vlm.paligemma.prompts import BUILDING_CLASSES, build_prompt, build_target


# HELPERS
def median_prediction(cfg: PaliGemmaConfig) -> str:
    if cfg.task == PaliGemmaTask.BUILDING_CLASS:
        idx = len(BUILDING_CLASSES) // 2
        return list(BUILDING_CLASSES.keys())[idx]
    else:
        jsonl = read_jsonl(cfg.base_dir / "all.jsonl")
        years = np.array(
            [
                record["buildings"][0]["attributes"][EubuccoField.construction_year]
                for record in jsonl
            ]
        )
        median = np.median(years)

        return (
            str((median // 10) * 10)
            if cfg.task == PaliGemmaTask.BUILDING_DECADE
            else str(int(median))
        )


def main(
    config: Path = typer.Option(
        Path("configs/predict/pg2-3b-mix-448.zeroshot.yaml"),
        "--config",
        "-c",
        exists=True,
        dir_okay=False,
        file_okay=True,
        readable=True,
        help="Path to predict config YAML.",
    ),
) -> None:
    cfg = load_paligemma_config(config)

    # Get the median value prediction
    median = median_prediction(cfg)
    print(median)

    # Create predictions
    records = read_jsonl(cfg.data.test_jsonl)
    predictions: list[dict[str, Any]] = []

    for record in records:
        predictions.append(
            {
                "id": record.get("id"),
                "task": str(cfg.task),
                "prompt": build_prompt(record, cfg.task),
                "target": build_target(record, cfg.task),
                "prediction": median,
            }
        )

    # Write results
    out_path = Path("outputs/predict/median/predictions.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(predictions, out_path)


if __name__ == "__main__":
    typer.run(main)
