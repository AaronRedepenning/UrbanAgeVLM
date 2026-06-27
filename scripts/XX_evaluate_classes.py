from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import typer
from sklearn.metrics import confusion_matrix

from urban_vlm.dataset.jsonl import read_jsonl
from urban_vlm.paligemma import load_paligemma_config
from urban_vlm.paligemma.prompts import BUILDING_CLASSES


# HELPERS
def compute_confusion_matrices(y_true, y_pred):
    class_order = list(BUILDING_CLASSES.keys())

    cm_counts = confusion_matrix(
        y_true,
        y_pred,
        labels=class_order,
    )

    row_sums = cm_counts.sum(axis=1, keepdims=True)

    cm_percent = (
        np.divide(
            cm_counts,
            row_sums,
            out=np.zeros_like(cm_counts, dtype=float),
            where=row_sums != 0,
        )
        * 100
    )

    cm_counts_df = pd.DataFrame(
        cm_counts,
        index=class_order,
        columns=class_order,
    )

    cm_percent_df = pd.DataFrame(
        cm_percent,
        index=class_order,
        columns=class_order,
    )

    return cm_counts_df, cm_percent_df


def plot_confusion_matrix(cm_percent_df, title, output_path=None):
    plt.figure(figsize=(9, 7))

    ax = sns.heatmap(
        cm_percent_df,
        annot=True,
        fmt=".0f",
        cmap="Blues",
        vmin=0,
        vmax=100,
        cbar_kws={"label": "Percentage of true class"},
        linewidths=0.5,
        linecolor="white",
    )

    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title(title)

    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    if output_path is not None:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")

    plt.show()


# MAIN
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
    prediction_jsonl = (
        Path("outputs/predict/building_class/adaptive_640/median/predictions.jsonl")
        # cfg.generation.output_jsonl
    )

    predictions = read_jsonl(
        prediction_jsonl,
    )
    y_true = [prediction["target"] for prediction in predictions]
    y_pred = [prediction["prediction"] for prediction in predictions]

    cm_counts_df, cm_percent_df = compute_confusion_matrices(y_true, y_pred)

    plot_confusion_matrix(
        cm_percent_df,
        title="Confusion Matrix: Building Construction Period",
        output_path=prediction_jsonl.parent / "confusion_matrix_percent.png",
    )


if __name__ == "__main__":
    typer.run(main)
