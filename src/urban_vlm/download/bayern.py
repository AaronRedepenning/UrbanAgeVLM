from pathlib import Path
from urllib.parse import unquote, urlparse

from urban_vlm.download.meta4 import download_meta4


def make_meta4_filename(url: str):
    return Path(unquote(urlparse(url).path)).name


def download_bayern(cfg: dict) -> None:
    bayern_cfg = cfg["bayern"]
    output_dir = Path(bayern_cfg["out_dir"])
    download_cfg = cfg.get("download", {})

    overwrite = download_cfg.get("overwrite", False)
    timeout_seconds = download_cfg.get("timeout_seconds", 60)

    for meta4_url in bayern_cfg["meta4_urls"]:
        meta4_path = output_dir / make_meta4_filename(meta4_url)

        download_meta4(
            meta4_url,
            meta4_path=meta4_path,
            output_dir=output_dir,
            overwrite=overwrite,
            timeout_seconds=timeout_seconds,
        )
