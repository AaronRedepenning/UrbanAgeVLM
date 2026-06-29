from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import typer

from urban_vlm.dataset.jsonl import read_jsonl
from urban_vlm.eubucco.schema import EubuccoField
from urban_vlm.paligemma import PaliGemmaConfig, load_paligemma_config
from urban_vlm.paligemma.config import PaliGemmaTask
from urban_vlm.paligemma.prompts import build_prompt, build_target


# HELPERS
def constant_baseline_prediction(cfg: PaliGemmaConfig) -> str:
    jsonl_file = cfg.data.predict_jsonl or cfg.data.test_jsonl
    jsonl = read_jsonl(jsonl_file)

    if cfg.task == PaliGemmaTask.BUILDING_CLASS:
        labels = [build_target(record, cfg.task) for record in jsonl]
        return Counter(labels).most_common(1)[0][0]

    years = np.array(
        [
            record["buildings"][0]["attributes"][EubuccoField.construction_year]
            for record in jsonl
        ]
    )

    median = np.median(years)

    if cfg.task == PaliGemmaTask.BUILDING_DECADE:
        return str(int((median // 10) * 10))

    return str(int(median))


def main(
    config: Path = typer.Option(
        Path("configs/predict/X__baseline.yaml"),
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
    baseline = constant_baseline_prediction(cfg)
    print(f"Baseline prediction: {baseline}")

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
                "prediction": baseline,
            }
        )

    # Write results
    out_path = cfg.generation.output_csv
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(predictions).to_csv(out_path)


if __name__ == "__main__":
    typer.run(main)
