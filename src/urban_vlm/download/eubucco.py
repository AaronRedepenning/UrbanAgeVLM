from pathlib import Path

from urban_vlm.download.config import DownloadOptionsConfig, EubuccoDownloadConfig
from urban_vlm.download.http import download_urls

TEMPLATE_URL = "https://s3.eubucco.com/eubucco/{version}/buildings/parquet/nuts_id={nuts_id}/{nuts_id}.parquet"


def make_eubucco_url(nuts_id: str, version: str) -> str:
    return TEMPLATE_URL.format(
        version=version,
        nuts_id=nuts_id.strip().upper(),
    )


def download_eubucco(
    cfg: EubuccoDownloadConfig, download_options: DownloadOptionsConfig
) -> list[Path]:
    urls = [make_eubucco_url(nuts_id, cfg.version) for nuts_id in cfg.nuts_ids]

    return download_urls(
        urls,
        output_dir=cfg.out_dir,
        overwrite=download_options.overwrite,
        timeout_seconds=download_options.timeout_seconds,
        max_workers=download_options.max_workers,
        show_progress=download_options.show_progress,
        overall_description=f"Downloading EUBUCCO {cfg.version}",
    )
