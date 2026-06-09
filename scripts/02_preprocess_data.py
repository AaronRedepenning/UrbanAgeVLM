from pathlib import Path

from urban_vlm.config import load_config
from urban_vlm.preprocess.manager import preprocess_buildings_and_match_tiles


def main(
    config: Path = Path("configs/preprocess.yaml"),
) -> None:
    cfg = load_config(config)
    preprocess_buildings_and_match_tiles(cfg)


if __name__ == "__main__":
    main()
