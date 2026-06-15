import xml.etree.ElementTree as ET
from pathlib import Path

from urban_vlm.download.http import download_file, download_urls


def _parse_meta4_urls(meta4_path: Path) -> list[str]:
    tree = ET.parse(meta4_path)
    root = tree.getroot()

    ns = {"m": "urn:ietf:params:xml:ns:metalink"}

    urls: list[str] = []

    for file_el in root.findall("m:file", ns):
        file_urls = [
            url_el.text.strip()
            for url_el in file_el.findall("m:url", ns)
            if url_el.text and url_el.text.strip()
        ]

        if not file_urls:
            name = file_el.attrib.get("name", "<unknown>")
            raise ValueError(f"No URLs found for file in metalink: {name}")

        # In metalink files, multiple URLs for one file are usually mirrors.
        # Keep this simple: use the first URL for each file.
        urls.append(file_urls[0])

    return urls


def download_meta4(
    meta4_url: str,
    output_dir: Path,
    *,
    overwrite: bool = False,
    timeout_seconds: int = 60,
    max_workers: int = 8,
    show_progress: bool = True,
) -> list[Path]:
    meta4_path = download_file(
        meta4_url,
        output_dir,
        overwrite=overwrite,
        timeout_seconds=timeout_seconds,
    )

    urls = _parse_meta4_urls(meta4_path)

    return download_urls(
        urls,
        output_dir,
        overwrite=overwrite,
        timeout_seconds=timeout_seconds,
        max_workers=max_workers,
        show_progress=show_progress,
        overall_description=f"Downloading {meta4_path.name}",
    )
