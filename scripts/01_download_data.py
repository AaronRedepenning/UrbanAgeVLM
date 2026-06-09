from pathlib import Path

from urban_vlm.config import load_config
from urban_vlm.download.manager import download_all


def main(
    config: Path = Path("configs/download.yaml"),
) -> None:
    cfg = load_config(config)
    download_all(cfg)


if __name__ == "__main__":
    main()
