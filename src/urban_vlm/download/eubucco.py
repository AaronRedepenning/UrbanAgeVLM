from pathlib import Path
from urllib.parse import unquote, urlparse

from urban_vlm.download.http import download_file

TEMPLATE_URL = "https://s3.eubucco.com/eubucco/{version}/buildings/parquet/nuts_id={nuts_id}/{nuts_id}.parquet"


def make_eubucco_url(nuts_id: str, version: str) -> str:
    return TEMPLATE_URL.format(
        version=version,
        nuts_id=nuts_id.strip().upper(),
    )


def make_eubucco_filename(url: str):
    return Path(unquote(urlparse(url).path)).name


def download_eubucco(cfg: dict) -> None:
    eubucco_cfg = cfg["eubucco"]
    out_dir = Path(eubucco_cfg["out_dir"])
    download_cfg = cfg.get("download", {})

    overwrite = download_cfg.get("overwrite", False)
    timeout_seconds = download_cfg.get("timeout_seconds", 60)

    version = eubucco_cfg.get("version", "v0.2")
    nuts_ids = eubucco_cfg.get("nuts_ids", [])

    if not nuts_ids:
        raise ValueError("No EUBUCCO nuts_ids specified!")

    for nuts_id in nuts_ids:
        url = make_eubucco_url(nuts_id, version)
        name = make_eubucco_filename(url)
        output_path = out_dir / name

        download_file(
            url,
            output_path,
            overwrite=overwrite,
            timeout_seconds=timeout_seconds,
        )
