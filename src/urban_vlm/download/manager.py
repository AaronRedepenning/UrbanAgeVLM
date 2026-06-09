from urban_vlm.download.bayern import download_bayern
from urban_vlm.download.eubucco import download_eubucco


def download_all(cfg: dict) -> None:
    if cfg.get("eubucco", {}).get("enabled", False):
        download_eubucco(cfg)

    if cfg.get("bayern", {}).get("enabled", False):
        download_bayern(cfg)
