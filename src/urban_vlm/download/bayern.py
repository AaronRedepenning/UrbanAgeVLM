from pathlib import Path

from urban_vlm.download.config import BayernDownloadConfig, DownloadOptionsConfig
from urban_vlm.download.meta4 import download_meta4
from urban_vlm.utils import url_filename


def download_bayern(
    cfg: BayernDownloadConfig, *, download_options: DownloadOptionsConfig
) -> None:
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for meta4_url in cfg.meta4_urls:
        meta4_url_str = str(meta4_url)
        meta4_path = out_dir / url_filename(meta4_url_str)

        download_meta4(
            meta4_url_str,
            meta4_path=meta4_path,
            output_dir=out_dir,
            overwrite=download_options.overwrite,
            timeout_seconds=download_options.timeout_seconds,
        )
