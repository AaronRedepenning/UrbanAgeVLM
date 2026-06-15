from pathlib import Path

from urban_vlm.download.config import DownloadOptionsConfig, NutsDownloadConfig
from urban_vlm.download.http import download_file


def download_nuts(
    cfg: NutsDownloadConfig, download_options: DownloadOptionsConfig
) -> list[Path]:
    return [
        download_file(
            str(cfg.url),
            cfg.out_dir,
            overwrite=download_options.overwrite,
            timeout_seconds=download_options.timeout_seconds,
        )
    ]
