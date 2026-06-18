from pathlib import Path

from urban_vlm.download.config import BayernDownloadConfig, DownloadOptionsConfig
from urban_vlm.download.meta4 import download_meta4
from urban_vlm.utils import url_filename


def download_bayern(
    cfg: BayernDownloadConfig, download_options: DownloadOptionsConfig
) -> list[Path]:
    downloaded_paths: list[Path] = []

    for meta4_url in cfg.meta4_urls:
        output_dir = cfg.out_dir / Path(url_filename(str(meta4_url))).stem
        downloaded_paths.extend(
            download_meta4(
                str(meta4_url),
                output_dir=output_dir,
                overwrite=download_options.overwrite,
                timeout_seconds=download_options.timeout_seconds,
                max_workers=download_options.max_workers,
                show_progress=download_options.show_progress,
            )
        )

    return downloaded_paths
