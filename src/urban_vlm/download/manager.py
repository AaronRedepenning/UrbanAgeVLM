from urban_vlm.download.bayern import download_bayern
from urban_vlm.download.config import DownloadConfig
from urban_vlm.download.eubucco import download_eubucco
from urban_vlm.download.nuts import download_nuts


def download_all(cfg: DownloadConfig) -> None:
    if cfg.nuts.enabled:
        download_nuts(cfg.nuts, download_options=cfg.download)

    if cfg.eubucco.enabled:
        download_eubucco(cfg.eubucco, download_options=cfg.download)

    if cfg.bayern.enabled:
        download_bayern(cfg.bayern, download_options=cfg.download)
