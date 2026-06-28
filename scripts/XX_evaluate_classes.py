from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import typer
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from urban_vlm.dataset.jsonl import read_jsonl
from urban_vlm.paligemma import load_paligemma_config
from urban_vlm.paligemma.prompts import BUILDING_CLASSES

# CONSTANTS
CLASS_ORDER = list(BUILDING_CLASSES.keys())


# HELPERS
def compute_confusion_matrices(y_true, y_pred):
    cm_counts = confusion_matrix(
        y_true,
        y_pred,
        labels=CLASS_ORDER,
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
        index=CLASS_ORDER,
        columns=CLASS_ORDER,
    )

    cm_percent_df = pd.DataFrame(
        cm_percent,
        index=CLASS_ORDER,
        columns=CLASS_ORDER,
    )

    return cm_counts_df, cm_percent_df


def add_support_to_labels(cm_counts_df):
    supports = cm_counts_df.sum(axis=1)

    labels_with_support = [
        f"{label}\n(n={supports[label]})" for label in cm_counts_df.index
    ]

    return labels_with_support


def plot_confusion_matrix(cm_counts_df, cm_percent_df, title, output_path=None):
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

    y_labels = add_support_to_labels(cm_counts_df)
    ax.set_yticklabels(y_labels, rotation=0)

    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title(title)

    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    if output_path is not None:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")


def plot_per_class_f1(report, output_path=None):
    rows = []

    for label in CLASS_ORDER:
        rows.append(
            {
                "class": label,
                "f1": report[label]["f1-score"],
                "support": report[label]["support"],
            }
        )

    df = pd.DataFrame(rows)

    plt.figure(figsize=(9, 5))
    bars = plt.bar(df["class"], df["f1"])

    plt.ylim(0, 1)
    plt.ylabel("F1 score")
    plt.xlabel("Construction period")
    plt.title("Per-Class F1 Score")

    plt.xticks(rotation=45, ha="right")

    for bar, support in zip(bars, df["support"]):
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.02,
            f"n={int(support)}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")


# MAIN
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
    cfg = load_paligemma_config(config)
    prediction_jsonl = (
        # Path("outputs/predict/building_class/adaptive_640/median/predictions.jsonl")
        cfg.generation.output_jsonl
    )

    predictions = read_jsonl(
        prediction_jsonl,
    )
    y_true = [prediction["target"] for prediction in predictions]
    y_pred = [prediction["prediction"] for prediction in predictions]

    # Compute confusion matrix
    cm_counts_df, cm_percent_df = compute_confusion_matrices(y_true, y_pred)
    cm_counts_df.to_csv(prediction_jsonl.parent / "confusion_matrix_counts.csv")
    cm_percent_df.to_csv(prediction_jsonl.parent / "confusion_matrix_percent.csv")

    plot_confusion_matrix(
        cm_counts_df,
        cm_percent_df,
        title="Confusion Matrix: Building Construction Period",
        output_path=prediction_jsonl.parent / "confusion_matrix_percent.png",
    )

    # Compute F1 scores
    macro_f1 = f1_score(
        y_true,
        y_pred,
        labels=CLASS_ORDER,
        average="macro",
        zero_division=0,
    )

    weighted_f1 = f1_score(
        y_true,
        y_pred,
        labels=CLASS_ORDER,
        average="weighted",
        zero_division=0,
    )

    print("Macro F1:", macro_f1)
    print("Weighted F1:", weighted_f1)

    report = classification_report(
        y_true,
        y_pred,
        labels=CLASS_ORDER,
        output_dict=True,
        zero_division=0,
    )

    plot_per_class_f1(report, prediction_jsonl.parent / "per_class_f1.png")

    # Show ALL plots
    plt.show()


if __name__ == "__main__":
    typer.run(main)
