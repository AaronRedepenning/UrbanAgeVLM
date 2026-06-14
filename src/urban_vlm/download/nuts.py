from pathlib import Path

from urban_vlm.download.config import DownloadOptionsConfig, NutsDownloadConfig
from urban_vlm.download.http import download_file
from urban_vlm.utils import url_filename


def download_nuts(
    cfg: NutsDownloadConfig, *, download_options: DownloadOptionsConfig
) -> None:
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    url = str(cfg.url)
    output_path = out_dir / url_filename(url)

    download_file(
        url,
        output_path,
        overwrite=download_options.overwrite,
        timeout_seconds=download_options.timeout_seconds,
    )
