import json
import logging
import random
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from urban_vlm.dataset.crops import save_crop_debug_image
from urban_vlm.dataset.jsonl import read_jsonl

logger = logging.getLogger(__name__)


def explore_jsonl_dataset(
    input_jsonl: str | Path,
    *,
    report_json: str | Path,
    figures_dir: str | Path,
    samples_dir: str | Path,
    n_samples: int = 25,
    seed: int = 42,
    construction_year_bin_width: int = 10,
) -> dict[str, Any]:
    input_jsonl = Path(input_jsonl)
    report_json = Path(report_json)
    figures_dir = Path(figures_dir)
    samples_dir = Path(samples_dir)

    figures_dir.mkdir(parents=True, exist_ok=True)
    samples_dir.mkdir(parents=True, exist_ok=True)
    report_json.parent.mkdir(parents=True, exist_ok=True)

    records = read_jsonl(input_jsonl)

    if not records:
        raise ValueError(f"No records found in {input_jsonl}")

    df = records_to_dataframe(records)

    summary = summarize_dataset(df, records)

    with report_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    plot_construction_year_histogram(
        df,
        figures_dir / "construction_year_histogram.png",
        bin_width=construction_year_bin_width,
    )

    plot_crop_size_scatter(
        df,
        figures_dir / "crop_size_scatter.png",
    )

    plot_crop_area_histogram(
        df,
        figures_dir / "crop_area_histogram.png",
    )

    save_sample_debug_images(
        records,
        samples_dir=samples_dir,
        n_samples=n_samples,
        seed=seed,
    )

    logger.info("Wrote dataset summary to %s.", report_json)
    logger.info("Wrote figures to %s.", figures_dir)

    return summary


def records_to_dataframe(records: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []

    for record in records:
        building = record["buildings"][0]
        attrs = building.get("attributes", {})
        crop = record.get("crop", {})

        crop_width = crop.get("width")
        crop_height = crop.get("height")

        bbox = building.get("geometry", {}).get("bbox")
        bbox_width = None
        bbox_height = None

        if bbox is not None and len(bbox) == 4:
            bbox_width = bbox[2] - bbox[0]
            bbox_height = bbox[3] - bbox[1]

        rows.append(
            {
                "id": record.get("id"),
                "image": record.get("image"),
                "crop_type": crop.get("type"),
                "crop_width": crop_width,
                "crop_height": crop_height,
                "crop_area": (
                    crop_width * crop_height
                    if crop_width is not None and crop_height is not None
                    else None
                ),
                "building_id": building.get("building_id"),
                "construction_year": attrs.get("construction_year"),
                "height": attrs.get("height"),
                "floors": attrs.get("floors"),
                "type": attrs.get("type"),
                "subtype": attrs.get("subtype"),
                "bbox_width": bbox_width,
                "bbox_height": bbox_height,
                "bbox_area": (
                    bbox_width * bbox_height
                    if bbox_width is not None and bbox_height is not None
                    else None
                ),
            }
        )

    return pd.DataFrame(rows)


def summarize_dataset(
    df: pd.DataFrame,
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    year = pd.to_numeric(df["construction_year"], errors="coerce")
    crop_area = pd.to_numeric(df["crop_area"], errors="coerce")
    crop_width = pd.to_numeric(df["crop_width"], errors="coerce")
    crop_height = pd.to_numeric(df["crop_height"], errors="coerce")

    summary = {
        "num_records": int(len(df)),
        "num_unique_images": int(df["image"].nunique()),
        "num_unique_buildings": int(df["building_id"].nunique()),
        "construction_year": {
            "count": int(year.notna().sum()),
            "missing": int(year.isna().sum()),
            "min": int(year.min()) if year.notna().any() else None,
            "max": int(year.max()) if year.notna().any() else None,
            "mean": float(year.mean()) if year.notna().any() else None,
            "median": float(year.median()) if year.notna().any() else None,
        },
        "crop": {
            "width_min": float(crop_width.min()) if crop_width.notna().any() else None,
            "width_max": float(crop_width.max()) if crop_width.notna().any() else None,
            "height_min": (
                float(crop_height.min()) if crop_height.notna().any() else None
            ),
            "height_max": (
                float(crop_height.max()) if crop_height.notna().any() else None
            ),
            "area_min": float(crop_area.min()) if crop_area.notna().any() else None,
            "area_max": float(crop_area.max()) if crop_area.notna().any() else None,
            "area_median": (
                float(crop_area.median()) if crop_area.notna().any() else None
            ),
        },
        "building_type_counts": _top_counts(df["type"]),
        "building_subtype_counts": _top_counts(df["subtype"]),
        "quality_checks": quality_checks(df, records),
    }

    return summary


def _top_counts(series: pd.Series, n: int = 20) -> dict[str, int]:
    counts = Counter(series.dropna().astype(str))
    return dict(counts.most_common(n))


def quality_checks(
    df: pd.DataFrame,
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "duplicate_record_ids": int(df["id"].duplicated().sum()),
        "duplicate_building_ids": int(df["building_id"].duplicated().sum()),
        "missing_image": int(df["image"].isna().sum()),
        "missing_construction_year": int(df["construction_year"].isna().sum()),
        "invalid_crop_dimensions": int(
            ((df["crop_width"] <= 0) | (df["crop_height"] <= 0)).sum()
        ),
        "bbox_outside_crop": count_bboxes_outside_crop(records),
        "footprint_points_outside_crop": count_records_with_footprint_outside_crop(
            records
        ),
    }


def count_bboxes_outside_crop(records: list[dict[str, Any]]) -> int:
    count = 0

    for record in records:
        crop = record["crop"]
        width = crop["width"]
        height = crop["height"]

        for building in record["buildings"]:
            bbox = building["geometry"]["bbox"]
            x_min, y_min, x_max, y_max = bbox

            if x_min < 0 or y_min < 0 or x_max > width or y_max > height:
                count += 1
                break

    return count


def count_records_with_footprint_outside_crop(records: list[dict[str, Any]]) -> int:
    count = 0

    for record in records:
        width = record["crop"]["width"]
        height = record["crop"]["height"]
        outside = False

        for building in record["buildings"]:
            rings = building["geometry"]["footprint"]

            for ring in rings:
                for x, y in ring:
                    if x < 0 or y < 0 or x > width or y > height:
                        outside = True
                        break

                if outside:
                    break

            if outside:
                break

        if outside:
            count += 1

    return count


def plot_construction_year_histogram(
    df: pd.DataFrame,
    output_path: str | Path,
    *,
    bin_width: int = 10,
) -> None:
    output_path = Path(output_path)

    years = pd.to_numeric(df["construction_year"], errors="coerce").dropna()

    if years.empty:
        logger.warning("No construction years available for histogram.")
        return

    min_year = int(years.min() // bin_width * bin_width)
    max_year = int((years.max() // bin_width + 1) * bin_width)
    bins = list(range(min_year, max_year + bin_width, bin_width))

    plt.figure(figsize=(12, 6))
    plt.hist(years, bins=bins)
    plt.xlabel("Construction year")
    plt.ylabel("Number of buildings")
    plt.title("Construction year distribution")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_crop_size_scatter(
    df: pd.DataFrame,
    output_path: str | Path,
) -> None:
    output_path = Path(output_path)

    plot_df = df.dropna(subset=["crop_width", "crop_height"])

    if plot_df.empty:
        logger.warning("No crop dimensions available for scatter plot.")
        return

    plt.figure(figsize=(7, 7))
    plt.scatter(plot_df["crop_width"], plot_df["crop_height"], alpha=0.25, s=8)
    plt.xlabel("Crop width, px")
    plt.ylabel("Crop height, px")
    plt.title("Crop dimensions")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_crop_area_histogram(
    df: pd.DataFrame,
    output_path: str | Path,
) -> None:
    output_path = Path(output_path)

    area = pd.to_numeric(df["crop_area"], errors="coerce").dropna()

    if area.empty:
        logger.warning("No crop area values available.")
        return

    plt.figure(figsize=(10, 6))
    plt.hist(area, bins=50)
    plt.xlabel("Crop area, px²")
    plt.ylabel("Number of records")
    plt.title("Crop area distribution")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def save_sample_debug_images(
    records: list[dict[str, Any]],
    *,
    samples_dir: Path,
    n_samples: int,
    seed: int,
) -> None:
    rng = random.Random(seed)

    sample_records = records.copy()
    rng.shuffle(sample_records)
    sample_records = sample_records[: min(n_samples, len(sample_records))]

    for i, record in enumerate(sample_records):
        record_id = record.get("id", f"record_{i}")
        output_path = samples_dir / f"{i:04d}_{record_id}.png"

        try:
            save_crop_debug_image(record, output_path)
        except Exception as exc:
            logger.warning(
                "Failed to render sample crop for record %s: %s",
                record_id,
                exc,
            )
