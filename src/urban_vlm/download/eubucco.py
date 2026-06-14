from pathlib import Path

from urban_vlm.download.config import DownloadOptionsConfig, EubuccoDownloadConfig
from urban_vlm.download.http import download_file
from urban_vlm.utils import url_filename

TEMPLATE_URL = "https://s3.eubucco.com/eubucco/{version}/buildings/parquet/nuts_id={nuts_id}/{nuts_id}.parquet"


def make_eubucco_url(nuts_id: str, version: str) -> str:
    return TEMPLATE_URL.format(
        version=version,
        nuts_id=nuts_id.strip().upper(),
    )


def download_eubucco(
    cfg: EubuccoDownloadConfig, *, download_options: DownloadOptionsConfig
) -> None:
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for nuts_id in cfg.nuts_ids:
        url = make_eubucco_url(nuts_id, cfg.version)
        output_path = out_dir / url_filename(url)

        download_file(
            url,
            output_path,
            overwrite=download_options.overwrite,
            timeout_seconds=download_options.timeout_seconds,
        )
