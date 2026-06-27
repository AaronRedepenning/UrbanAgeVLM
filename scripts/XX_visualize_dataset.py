from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import typer

from urban_vlm.dataset import load_prepare_config
from urban_vlm.dataset.jsonl import read_jsonl
from urban_vlm.eubucco.schema import EubuccoField


# HELPERS
def style_ax(ax, title, xlabel="Construction year", ylabel="Number of buildings"):
    ax.set_title(title, fontsize=14, weight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def buildings_jsonl_to_dataframe(path: Path, attributes: list[str]) -> pd.DataFrame:
    jsonl = read_jsonl(path)

    # Only keep attributes we care about
    buildings = {
        attribute: [record["buildings"][0]["attributes"][attribute] for record in jsonl]
        for attribute in attributes
    }

    return pd.DataFrame.from_dict(buildings)


def create_histograms(
    buildings: pd.DataFrame, *, out_dir: Path | None = None, show: bool = True
) -> None:
    N = len(buildings)

    subtypes = sorted(buildings[EubuccoField.subtype].unique())
    years = [
        buildings.loc[
            buildings[EubuccoField.subtype] == subtype, EubuccoField.construction_year
        ]
        for subtype in subtypes
    ]

    cmap = plt.get_cmap("tab20")
    colors = [cmap(i) for i in range(len(subtypes))]

    # 1. Histogram of building construction year
    fig, ax = plt.subplots(figsize=(11, 5))

    ax.hist(
        years,
        bins=40,
        stacked=True,
        label=subtypes,
        color=colors,
        edgecolor="white",
        linewidth=0.5,
    )

    ax.text(
        0.99,
        0.98,
        f"n = {N:,} buildings",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
    )

    style_ax(ax, "Construction Years by Building Type")

    ax.legend(
        title="Building type", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False
    )

    plt.tight_layout()
    if out_dir:
        plt.savefig(out_dir / "year_histogram.png")

    # 1. Histogram of building construction decade
    start_decade = (buildings[EubuccoField.construction_year].min() // 10) * 10
    end_decade = (buildings[EubuccoField.construction_year].max() // 10) * 10 + 10

    decade_bins = np.arange(start_decade, end_decade + 10, 10)

    fig, ax = plt.subplots(figsize=(11, 5))

    ax.hist(
        years,
        bins=decade_bins,
        stacked=True,
        label=subtypes,
        color=colors,
        edgecolor="white",
        linewidth=0.7,
    )

    ax.text(
        0.99,
        0.98,
        f"n = {N:,} buildings",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
    )

    ax.set_xticks(decade_bins[::2])
    ax.tick_params(axis="x", rotation=45)

    style_ax(ax, "Buildings by Construction Decade and Type")

    ax.legend(
        title="Building type", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False
    )

    plt.tight_layout()
    if out_dir:
        plt.savefig(out_dir / "decade_histogram.png")

    # 0. Histogram of building construction classes
    class_bins = [-np.inf, 1799, 1850, 1900, 1950, 2000, 2015, np.inf]

    class_labels = [
        "Before 1800",
        "1800-1850",
        "1851-1900",
        "1901-1950",
        "1951-2000",
        "2001-2015",
        "After 2015",
    ]

    buildings["year_class"] = pd.cut(
        buildings[EubuccoField.construction_year],
        bins=class_bins,
        labels=class_labels,
        right=True,
    )

    class_type_counts = pd.crosstab(
        buildings["year_class"], buildings[EubuccoField.subtype]
    ).reindex(class_labels)

    fig, ax = plt.subplots(figsize=(11, 5))

    class_type_counts.plot(
        kind="bar",
        stacked=True,
        ax=ax,
        color=colors[: len(class_type_counts.columns)],
        edgecolor="white",
        linewidth=0.7,
    )

    ax.text(
        0.99,
        0.98,
        f"n = {N:,} buildings",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
    )

    style_ax(
        ax,
        "Buildings by Construction Year Class and Type",
        xlabel="Construction year class",
    )

    ax.tick_params(axis="x", rotation=35)

    ax.legend(
        title="Building type", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False
    )

    plt.tight_layout()
    if out_dir:
        plt.savefig(out_dir / "classes_histogram.png")


# MAIN
def main(
    config: Path = typer.Option(
        Path("configs/prepare.multi_640.yaml"),
        "--config",
        "-c",
        exists=True,
        dir_okay=False,
        file_okay=True,
        readable=True,
        help="Path to preprocess config YAML.",
    ),
) -> None:
    cfg = load_prepare_config(config)

    for jsonl_file in [
        cfg.outputs.all_jsonl,
        cfg.outputs.train_jsonl,
        cfg.outputs.val_jsonl,
        cfg.outputs.test_jsonl,
    ]:
        # Read buildings from JSONL
        buildings = buildings_jsonl_to_dataframe(
            cfg.base_dir / cfg.crops[0].name / jsonl_file,
            attributes=[EubuccoField.subtype, EubuccoField.construction_year],
        )

        # Create histograms
        out_dir = Path("outputs/dataset") / jsonl_file.stem
        out_dir.mkdir(parents=True, exist_ok=True)
        create_histograms(buildings, out_dir=out_dir, show=False)


if __name__ == "__main__":
    typer.run(main)
