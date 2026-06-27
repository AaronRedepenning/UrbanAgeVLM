from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import typer

from urban_vlm.dataset.jsonl import read_jsonl
from urban_vlm.preprocess import load_preprocess_config


# HELPERS
def style_ax(ax, title, xlabel="Construction year", ylabel="Number of buildings"):
    ax.set_title(title, fontsize=14, weight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# MAIN
def main(
    config: Path = typer.Option(
        Path("configs/preprocess.yaml"),
        "--config",
        "-c",
        exists=True,
        dir_okay=False,
        file_okay=True,
        readable=True,
        help="Path to preprocess config YAML.",
    ),
) -> None:
    # cfg = load_preprocess_config(config)
    type_col = "subtype"
    year_col = "construction_year"

    # Read building years
    buildings_jsonl = read_jsonl("data/processed/adaptive_640/all.jsonl")

    buildings = {
        type_col: [
            record["buildings"][0]["attributes"][type_col] for record in buildings_jsonl
        ],
        year_col: [
            record["buildings"][0]["attributes"][year_col] for record in buildings_jsonl
        ],
    }
    buildings = pd.DataFrame.from_dict(buildings)
    # buildings = gpd.read_parquet(cfg.outputs.cleaned_buildings_file)

    N = len(buildings)
    median_year = buildings[year_col].median()

    subtypes = sorted(buildings[type_col].unique())
    years = [
        buildings.loc[buildings[type_col] == subtype, year_col] for subtype in subtypes
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
    plt.savefig("year_histogram.png")

    # 1. Histogram of building construction decade
    start_decade = (buildings[year_col].min() // 10) * 10
    end_decade = (buildings[year_col].max() // 10) * 10 + 10

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
    plt.savefig("decade_histogram.png")

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
        buildings[year_col], bins=class_bins, labels=class_labels, right=True
    )

    class_type_counts = pd.crosstab(
        buildings["year_class"], buildings[type_col]
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
    plt.savefig("classes_histogram.png")


if __name__ == "__main__":
    typer.run(main)
