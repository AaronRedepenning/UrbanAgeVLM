from pathlib import Path

from urban_vlm.config import load_config
from urban_vlm.dataset.manager import prepare_jsonl_dataset


def main(
    config: Path = Path("configs/prepare.yaml"),
) -> None:
    cfg = load_config(config)
    prepare_jsonl_dataset(cfg)


if __name__ == "__main__":
    main()
