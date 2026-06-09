from pathlib import Path

from urban_vlm.dataset.explore import explore_jsonl_dataset


def main() -> None:
    explore_jsonl_dataset(
        Path("data/processed/buildings.jsonl"),
        report_json=Path("outputs/reports/data_summary.json"),
        figures_dir=Path("outputs/figures"),
        samples_dir=Path("outputs/figures/sample_crops"),
        n_samples=25,
    )


if __name__ == "__main__":
    main()
