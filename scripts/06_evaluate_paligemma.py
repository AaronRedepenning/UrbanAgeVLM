import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import typer

from urban_vlm.paligemma import evaluate_prediction_jsonl


def read_jsonl(path: str | Path) -> pd.DataFrame:
    path = Path(path)

    with path.open("r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    return pd.DataFrame(records)


def main(
    config: Path = typer.Option(
        Path("configs/predict.yaml"),
        "--config",
        "-c",
        exists=True,
        dir_okay=False,
        file_okay=True,
        readable=True,
        help="Path to predict config YAML.",
    ),
) -> None:
    evaluate_prediction_jsonl(
        "outputs/adaptive_640-3_exp/predictions.jsonl",
        metrics_json="outputs/adaptive_640-3_exp/metrics.json",
        per_record_metrics_jsonl="outputs/adaptive_640-3_exp/per_record_metrics.jsonl",
    )

    metrics_path = Path("outputs/adaptive_640-3_exp/per_record_metrics.jsonl")

    df = read_jsonl(metrics_path)

    # Keep only rows with usable target + prediction.
    df = df[
        (df["status"] == "evaluated") & df["target"].notna() & df["prediction"].notna()
    ].copy()

    df["target"] = df["target"].astype(int)
    df["prediction"] = df["prediction"].astype(int)
    df["absolute_error"] = (df["prediction"] - df["target"]).abs()

    mae = df["absolute_error"].mean()
    rmse = np.sqrt(((df["prediction"] - df["target"]) ** 2).mean())

    # Add tiny jitter so repeated decade pairs do not fully overlap.
    rng = np.random.default_rng(42)
    jitter = 1.25

    x = df["target"] + rng.normal(0, jitter, size=len(df))
    y = df["prediction"] + rng.normal(0, jitter, size=len(df))

    min_decade = int(min(df["target"].min(), df["prediction"].min()))
    max_decade = int(max(df["target"].max(), df["prediction"].max()))

    # Round axis bounds to decades.
    min_decade = (min_decade // 10) * 10
    max_decade = ((max_decade + 9) // 10) * 10

    fig, ax = plt.subplots(figsize=(8, 8))

    ax.scatter(
        x,
        y,
        alpha=0.35,
        s=28,
        edgecolors="none",
    )

    # Perfect-prediction diagonal.
    ax.plot(
        [min_decade, max_decade],
        [min_decade, max_decade],
        linestyle="--",
        linewidth=1.5,
        label="Perfect prediction",
    )

    # Within 1 decade diagonal.
    ax.plot(
        [min_decade, max_decade],
        [min_decade + 10, max_decade + 10],
        linestyle="--",
        linewidth=1.5,
        c="gray",
        alpha=0.5,
    )
    ax.plot(
        [min_decade, max_decade],
        [min_decade - 10, max_decade - 10],
        linestyle="--",
        linewidth=1.5,
        label="±1 decade",
        c="gray",
        alpha=0.5,
    )

    ax.set_title(
        f"Predicted vs. true construction decade\n"
        f"n={len(df):,} · MAE={mae:.1f} years · RMSE={rmse:.1f} years"
    )
    ax.set_xlabel("True construction decade")
    ax.set_ylabel("Predicted construction decade")

    tick_step = 20
    ticks = list(range(min_decade, max_decade + 1, tick_step))

    ax.set_xticks(ticks)
    ax.set_yticks(ticks)

    ax.set_xlim(min_decade - 5, max_decade + 5)
    ax.set_ylim(min_decade - 5, max_decade + 5)

    ax.tick_params(axis="x", labelrotation=45)
    ax.tick_params(axis="both", labelsize=9)

    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)

    plt.tight_layout()
    plt.savefig("outputs/adaptive_640-3_exp/per_record_metrics.png")


if __name__ == "__main__":
    typer.run(main)
