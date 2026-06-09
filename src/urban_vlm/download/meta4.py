import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, urlparse

from urban_vlm.download.http import download_file


@dataclass(frozen=True)
class MetalinkFile:
    name: str
    urls: list[str]
    size: int | None = None
    hashes: dict[str, str] | None = None


def filename_from_url(url: str) -> str:
    return Path(unquote(urlparse(url).path)).name


def parse_meta4(meta4_path: Path) -> list[MetalinkFile]:
    tree = ET.parse(meta4_path)
    root = tree.getroot()

    # Metalink v4 namespace
    ns = {"m": "urn:ietf:params:xml:ns:metalink"}

    files: list[MetalinkFile] = []

    for file_el in root.findall("m:file", ns):
        name = file_el.attrib.get("name")

        urls = [
            url_el.text.strip()
            for url_el in file_el.findall("m:url", ns)
            if url_el.text and url_el.text.strip()
        ]

        if not name and urls:
            name = filename_from_url(urls[0])

        if not name:
            raise ValueError(f"Could not determine filename in {meta4_path}")

        size_el = file_el.find("m:size", ns)
        size = int(size_el.text) if size_el is not None and size_el.text else None

        hashes: dict[str, str] = {}
        for hash_el in file_el.findall("m:hash", ns):
            hash_type = hash_el.attrib.get("type")
            hash_value = hash_el.text.strip() if hash_el.text else None
            if hash_type and hash_value:
                hashes[hash_type.lower()] = hash_value

        files.append(
            MetalinkFile(
                name=name,
                urls=urls,
                size=size,
                hashes=hashes or None,
            )
        )

    return files


def download_meta4(
    meta4_url: str,
    *,
    meta4_path: Path,
    output_dir: Path,
    overwrite: bool = False,
    timeout_seconds: int = 60,
) -> list[Path]:
    meta4_path.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    download_file(
        meta4_url,
        meta4_path,
        overwrite=overwrite,
        timeout_seconds=timeout_seconds,
    )

    files = parse_meta4(meta4_path)

    downloaded_paths: list[Path] = []

    for item in files:
        if not item.urls:
            raise ValueError(f"No URLs found for file in metalink: {item.name}")

        output_path = output_dir / item.name

        # Try mirrors in order.
        last_error: Exception | None = None

        for url in item.urls:
            try:
                downloaded = download_file(
                    url,
                    output_path,
                    overwrite=overwrite,
                    timeout_seconds=timeout_seconds,
                )
                downloaded_paths.append(downloaded)
                last_error = None
                break
            except Exception as exc:
                last_error = exc

        if last_error is not None:
            raise RuntimeError(f"Failed to download {item.name}") from last_error

    return downloaded_paths
