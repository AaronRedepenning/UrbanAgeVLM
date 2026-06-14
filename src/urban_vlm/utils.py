from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import geopandas as gpd
import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)

    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if raw is None:
        raise ValueError(f"Config file is empty: {path}")

    if not isinstance(raw, dict):
        raise TypeError(f"Config file must contain a YAML mapping: {path}")

    return raw


def url_filename(url: str) -> str:
    return Path(unquote(urlparse(url).path)).name


def read_geodataframe(path: Path, *, columns: list[str]) -> gpd.GeoDataFrame:
    suffix = path.suffix.lower()

    if suffix in {".parquet", ".geoparquet"}:
        return gpd.read_parquet(path, columns=columns)

    if suffix in {".gpkg", ".geojson", ".json", ".shp"}:
        return gpd.read_file(path, columns=columns)

    raise ValueError(f"Unsupported input file type: {path.suffix}")


def write_geodataframe(
    gdf: gpd.GeoDataFrame,
    path: str | Path,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    suffix = path.suffix.lower()

    if suffix in {".parquet", ".geoparquet"}:
        gdf.to_parquet(path)
        return

    if suffix in {".gpkg"}:
        gdf.to_file(path, driver="GPKG")
        return

    if suffix in {".geojson", ".json"}:
        gdf.to_file(path, driver="GeoJSON")
        return

    raise ValueError(f"Unsupported output file type: {path.suffix}")
