from pathlib import Path

from urban_vlm.download import load_download_config
from urban_vlm.plots.mapping import make_coverage_map


def main(
    config: Path = Path("configs/download.yaml"),
) -> None:
    make_coverage_map(
        load_download_config(config),
        out_path=Path("/mnt/c/Users/aaron/OneDrive/Desktop/nuts-map.html"),
    )
    # explore_jsonl_dataset(
    #     Path("data/processed/single_buildings.jsonl"),
    #     report_json=Path("outputs/reports/data_summary.json"),
    #     figures_dir=Path("outputs/figures"),
    #     samples_dir=Path("outputs/figures/sample_crops"),
    #     n_samples=25,
    # )


if __name__ == "__main__":
    main()
