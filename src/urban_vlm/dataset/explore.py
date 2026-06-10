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

    write_sample_crop_groups(
        records,
        samples_dir=samples_dir,
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


def write_sample_crop_groups(
    records: list[dict[str, Any]],
    *,
    samples_dir: str | Path,
    n_per_group: int = 25,
    seed: int = 42,
) -> dict[str, int]:
    samples_dir = Path(samples_dir)
    samples_dir.mkdir(parents=True, exist_ok=True)

    df = records_to_dataframe(records)

    groups: dict[str, pd.DataFrame] = {
        "random": sample_random(df, n=n_per_group, seed=seed),
        "oldest": sample_oldest(df, n=n_per_group),
        "newest": sample_newest(df, n=n_per_group),
        "largest_crops": sample_largest_crops(df, n=n_per_group),
        "smallest_crops": sample_smallest_crops(df, n=n_per_group),
        # "clipped": sample_clipped(df, n=n_per_group, seed=seed),
        # "edge_cases": sample_edge_cases(df, n=n_per_group),
    }

    written_counts: dict[str, int] = {}

    record_lookup = {record["id"]: record for record in records}

    for group_name, group_df in groups.items():
        group_dir = samples_dir / group_name
        group_dir.mkdir(parents=True, exist_ok=True)

        count = 0

        for i, row in enumerate(group_df.itertuples(index=False)):
            record_id = row.id
            record = record_lookup.get(record_id)

            if record is None:
                continue

            filename = f"{i:04d}_{safe_filename(record_id)}.png"
            output_path = group_dir / filename

            title = make_sample_title(record, group_name=group_name)

            try:
                save_crop_debug_image(
                    record,
                    output_path,
                    title=title,
                )
                count += 1
            except Exception as exc:
                logger.warning(
                    "Failed to render sample crop %s in group %s: %s",
                    record_id,
                    group_name,
                    exc,
                )

        written_counts[group_name] = count
        logger.info("Wrote %s sample crops for group %s.", count, group_name)

    return written_counts


def sample_random(df: pd.DataFrame, *, n: int, seed: int) -> pd.DataFrame:
    if df.empty:
        return df

    return df.sample(
        n=min(n, len(df)),
        random_state=seed,
    )


def sample_oldest(df: pd.DataFrame, *, n: int) -> pd.DataFrame:
    valid = df.dropna(subset=["construction_year"])

    return valid.sort_values(
        ["construction_year", "id"],
        ascending=[True, True],
    ).head(n)


def sample_newest(df: pd.DataFrame, *, n: int) -> pd.DataFrame:
    valid = df.dropna(subset=["construction_year"])

    return valid.sort_values(
        ["construction_year", "id"],
        ascending=[False, True],
    ).head(n)


def sample_largest_crops(df: pd.DataFrame, *, n: int) -> pd.DataFrame:
    valid = df.dropna(subset=["crop_area"])

    return valid.sort_values(
        ["crop_area", "id"],
        ascending=[False, True],
    ).head(n)


def sample_smallest_crops(df: pd.DataFrame, *, n: int) -> pd.DataFrame:
    valid = df.dropna(subset=["crop_area"])

    valid = valid[valid["crop_area"] > 0]

    return valid.sort_values(
        ["crop_area", "id"],
        ascending=[True, True],
    ).head(n)


def sample_clipped(
    df: pd.DataFrame,
    *,
    n: int,
    seed: int,
) -> pd.DataFrame:
    clipped = df[df["crop_clipped"]].copy()

    if clipped.empty:
        return clipped

    return clipped.sample(
        n=min(n, len(clipped)),
        random_state=seed,
    )


def sample_edge_cases(df: pd.DataFrame, *, n: int) -> pd.DataFrame:
    """
    Pull examples likely to reveal bugs:
    - very small crops
    - very large crops
    - very old buildings
    - very new buildings
    - clipped crops
    """
    frames: list[pd.DataFrame] = []

    if not df.empty:
        frames.append(sample_smallest_crops(df, n=max(1, n // 5)))
        frames.append(sample_largest_crops(df, n=max(1, n // 5)))
        frames.append(sample_oldest(df, n=max(1, n // 5)))
        frames.append(sample_newest(df, n=max(1, n // 5)))
        frames.append(df[df["crop_clipped"]].head(max(1, n // 5)))

    if not frames:
        return df.head(0)

    out = pd.concat(frames, ignore_index=True)
    out = out.drop_duplicates(subset=["id"])

    return out.head(n)


def make_sample_title(
    record: dict[str, Any],
    *,
    group_name: str,
) -> str:
    building = record.get("buildings", [{}])[0]
    attrs = building.get("attributes", {})
    crop = record.get("crop", {})

    year = attrs.get("construction_year")
    decade = attrs.get("construction_decade_label") or attrs.get("construction_decade")
    building_type = attrs.get("type")
    subtype = attrs.get("subtype")

    parts = [
        group_name,
        f"id={record.get('id')}",
        f"year={year}",
        f"decade={decade}",
        f"type={building_type}",
        f"subtype={subtype}",
        f"crop={crop.get('width')}x{crop.get('height')}",
    ]

    if crop.get("clipped"):
        parts.append(f"clipped={','.join(crop.get('clip_sides', []))}")

    return " | ".join(str(part) for part in parts if part is not None)


def safe_filename(value: str) -> str:
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.")

    return "".join(char if char in allowed else "_" for char in value)
