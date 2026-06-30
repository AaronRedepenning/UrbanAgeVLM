from pathlib import Path

from urban_vlm.download.atom import download_atom
from urban_vlm.download.config import BerlinDownloadConfig, DownloadOptionsConfig


def download_berlin(
    cfg: BerlinDownloadConfig, download_options: DownloadOptionsConfig
) -> list[Path]:
    downloaded_paths = download_atom(
        str(cfg.atom_url),
        cfg.out_dir,
        extract_zip=True,
        extract_dir=cfg.out_dir,
        delete_zip=False,
        overwrite=True,  # download_options.overwrite,
        timeout_seconds=download_options.timeout_seconds,
        max_workers=download_options.max_workers,
        show_progress=download_options.show_progress,
    )

    return downloaded_paths
