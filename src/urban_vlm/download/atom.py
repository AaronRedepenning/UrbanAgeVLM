import re
import shutil
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import unquote, urljoin, urlparse
from zipfile import ZipFile

from urban_vlm.download.http import download_file, download_urls

ATOM_NS = {"a": "http://www.w3.org/2005/Atom"}

ATOM_TYPES = {
    "application/atom+xml",
}

DOWNLOAD_TYPES = {
    "application/zip",
    "application/x-zip-compressed",
    "image/tiff",
    "image/geotiff",
    "application/geotiff",
    "application/octet-stream",
}

DOWNLOAD_EXTENSIONS = (
    ".zip",
    ".tif",
    ".tiff",
    ".geotiff",
    ".jp2",
    ".jpg",
    ".jpeg",
)


def _normalise_mime_type(value: str | None) -> str:
    if not value:
        return ""
    return value.split(";", 1)[0].strip().lower()


def _url_filename(url: str) -> str:
    return unquote(Path(urlparse(url).path).name)


def _dedupe_keep_order(urls: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []

    for url in urls:
        if url not in seen:
            seen.add(url)
            result.append(url)

    return result


def _is_atom_link(href: str, mime_type: str) -> bool:
    path = urlparse(href).path.lower()
    return mime_type in ATOM_TYPES or path.endswith(".atom")


def _is_download_link(href: str, rel: str, mime_type: str) -> bool:
    if rel in {"self", "describedby", "up", "search"}:
        return False

    if _is_atom_link(href, mime_type):
        return False

    path = urlparse(href).path.lower()

    return (
        rel == "enclosure"
        or mime_type in DOWNLOAD_TYPES
        or path.endswith(DOWNLOAD_EXTENSIONS)
    )


def _parse_atom_links(atom_path: Path, base_url: str) -> tuple[list[str], list[str]]:
    tree = ET.parse(atom_path)
    root = tree.getroot()

    download_urls_: list[str] = []
    child_atom_urls: list[str] = []

    for link_el in root.findall(".//a:link", ATOM_NS):
        href = link_el.attrib.get("href")
        if not href:
            continue

        href = urljoin(base_url, href)
        rel = link_el.attrib.get("rel", "").strip().lower()
        mime_type = _normalise_mime_type(link_el.attrib.get("type"))

        if rel in {"self", "describedby"}:
            continue

        if _is_atom_link(href, mime_type):
            child_atom_urls.append(href)
        elif _is_download_link(href, rel, mime_type):
            download_urls_.append(href)

    return (
        _dedupe_keep_order(download_urls_),
        _dedupe_keep_order(child_atom_urls),
    )


def _extract_zip_with_python(zip_path: Path, extract_dir: Path) -> list[Path]:
    extract_dir.mkdir(parents=True, exist_ok=True)

    with ZipFile(zip_path) as zf:
        zf.extractall(extract_dir)

    return [p for p in extract_dir.rglob("*") if p.is_file()]


def _extract_zip_with_7z(zip_path: Path, extract_dir: Path) -> list[Path]:
    seven_zip = shutil.which("7z") or shutil.which("7za") or shutil.which("7zz")
    if seven_zip is None:
        raise RuntimeError(
            "Python zipfile could not extract this ZIP, and no 7z/7za/7zz "
            "executable was found. Install p7zip-full / 7zip."
        )

    extract_dir.mkdir(parents=True, exist_ok=True)

    subprocess.run(
        [
            seven_zip,
            "x",
            str(zip_path),
            f"-o{extract_dir}",
            "-y",
        ],
        check=True,
    )

    return [p for p in extract_dir.rglob("*") if p.is_file()]


def extract_zip_robust(zip_path: Path, extract_dir: Path) -> list[Path]:
    try:
        return _extract_zip_with_python(zip_path, extract_dir)
    except NotImplementedError:
        return _extract_zip_with_7z(zip_path, extract_dir)


def unzip_files(
    zip_paths: list[Path],
    extract_dir: Path,
    *,
    delete_zip: bool = False,
    overwrite: bool = False,
) -> list[Path]:
    import shutil

    extracted: list[Path] = []

    for zip_path in zip_paths:
        if zip_path.suffix.lower() != ".zip":
            continue

        tile_extract_dir = extract_dir / zip_path.stem

        if tile_extract_dir.exists() and overwrite:
            shutil.rmtree(tile_extract_dir)

        if tile_extract_dir.exists() and not overwrite:
            extracted.extend(p for p in tile_extract_dir.rglob("*") if p.is_file())
            continue

        extracted.extend(extract_zip_robust(zip_path, tile_extract_dir))

        if delete_zip:
            zip_path.unlink()

    return extracted


def list_atom_download_urls(
    atom_url: str,
    cache_dir: Path,
    *,
    overwrite_atom: bool = True,
    timeout_seconds: int = 60,
    filename_regex: str | None = None,
) -> list[str]:
    """
    Recursively parse an INSPIRE ATOM download service and return actual file URLs.

    For Berlin DOP, the root feed usually points to a dataset feed like 0.atom.
    That dataset feed then contains the actual imagery file links.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)

    seen_feeds: set[str] = set()
    found_downloads: list[str] = []

    pattern = re.compile(filename_regex) if filename_regex else None

    def visit(feed_url: str) -> None:
        if feed_url in seen_feeds:
            return

        seen_feeds.add(feed_url)

        atom_path = download_file(
            feed_url,
            cache_dir,
            overwrite=overwrite_atom,
            timeout_seconds=timeout_seconds,
        )

        downloads, child_feeds = _parse_atom_links(atom_path, feed_url)

        for url in downloads:
            filename = _url_filename(url)
            if pattern is None or pattern.search(filename):
                found_downloads.append(url)

        for child_feed_url in child_feeds:
            visit(child_feed_url)

    visit(atom_url)

    return _dedupe_keep_order(found_downloads)


def download_atom(
    atom_url: str,
    output_dir: Path,
    *,
    atom_cache_dir: Path | None = None,
    extract_zip: bool = True,
    extract_dir: Path | None = None,
    delete_zip: bool = False,
    overwrite: bool = False,
    timeout_seconds: int = 60,
    max_workers: int = 8,
    show_progress: bool = True,
    filename_regex: str | None = None,
) -> list[Path]:
    """
    Download all files referenced by an INSPIRE ATOM feed.

    If extract_zip=True, ZIP files are extracted automatically after download.
    Returns extracted paths when extracting, otherwise downloaded paths.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if atom_cache_dir is None:
        atom_cache_dir = output_dir / "_atom_feeds"

    urls = list_atom_download_urls(
        atom_url,
        atom_cache_dir,
        overwrite_atom=True,
        timeout_seconds=timeout_seconds,
        filename_regex=filename_regex,
    )

    if not urls:
        raise ValueError(f"No downloadable file URLs found in ATOM feed: {atom_url}")

    downloaded_paths = download_urls(
        urls,
        output_dir,
        overwrite=overwrite,
        timeout_seconds=timeout_seconds,
        max_workers=max_workers,
        show_progress=show_progress,
        overall_description=f"Downloading {len(urls)} files from ATOM feed",
    )

    if not extract_zip:
        return downloaded_paths

    if extract_dir is None:
        extract_dir = output_dir / "extracted"

    return unzip_files(
        downloaded_paths,
        extract_dir,
        delete_zip=delete_zip,
        overwrite=overwrite,
    )
