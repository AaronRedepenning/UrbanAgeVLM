import logging
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry.base import BaseGeometry

from urban_vlm.dataset.schema import BuildingField
from urban_vlm.eubucco.schema import (
    ALL_REQUIRED_FIELDS,
    EubuccoField,
    coerce_numeric_columns,
    require_columns,
)
from urban_vlm.utils import read_geodataframe

logger = logging.getLogger(__name__)


def read_eubucco_buildings(
    path: str | Path,
    *,
    aoi: BaseGeometry | gpd.GeoSeries | gpd.GeoDataFrame | str | Path | None = None,
    aoi_crs: str | None = None,
    target_crs: str = "EPSG:25832",
    dissolve_by_id: bool = False,
    require_construction_year: bool = False,
    min_area_m2: float | None = None,
    max_area_m2: float | None = None,
) -> gpd.GeoDataFrame:
    """
    Read EUBUCCO building data using the project's required EUBUCCO schema.

    This function:
    - reads only the standard required EUBUCCO columns,
    - reprojects to the target CRS,
    - coerces numeric columns,
    - removes invalid / empty geometries,
    - optionally filters to an area of interest,
    - optionally dissolves multipart EUBUCCO IDs into one building record,
    - adds a project-derived ``part_count`` column.

    Parameters
    ----------
    path:
        Path to an EUBUCCO file.
    aoi:
        Optional area of interest. Can be a shapely geometry, GeoSeries,
        GeoDataFrame, or path to a vector file.
    aoi_crs:
        CRS of the AOI if ``aoi`` is a raw shapely geometry or has no CRS.
    target_crs:
        CRS returned by this function.
    dissolve_by_id:
        If true, IDs ending in part suffixes like ``-1`` or ``-2`` are collapsed
        to the base ID and geometries are dissolved.
    min_area_m2
        Minumum allowd building area.
    max_area_m2
        Maximum allowed building area.

    Returns
    -------
    geopandas.GeoDataFrame
        Cleaned EUBUCCO buildings in ``target_crs``.
    """
    path = Path(path)

    buildings = read_geodataframe(
        path,
        columns=ALL_REQUIRED_FIELDS,
    )
    require_columns(buildings, ALL_REQUIRED_FIELDS)

    buildings = buildings.to_crs(target_crs)
    buildings = _clean_basic_eubucco(buildings)

    if aoi is not None:
        buildings = _filter_to_aoi(
            buildings,
            aoi=aoi,
            aoi_crs=aoi_crs,
            target_crs=target_crs,
        )

    if dissolve_by_id:
        buildings = _dissolve_building_parts_by_id(buildings)
    else:
        buildings = buildings.copy()
        buildings[str(BuildingField.PART_COUNT)] = 1

    if require_construction_year:
        buildings = buildings[buildings[EubuccoField.construction_year].notna()].copy()

    buildings = _filter_by_area(
        buildings, min_area_m2=min_area_m2, max_area_m2=max_area_m2
    )

    return buildings


def _clean_basic_eubucco(buildings: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    buildings = buildings.copy()
    buildings = coerce_numeric_columns(buildings)

    geometry_column = buildings.geometry.name

    before_count = len(buildings)

    buildings = buildings[~buildings[geometry_column].isna()].copy()
    buildings = buildings[~buildings.geometry.is_empty].copy()

    dropped_empty_count = before_count - len(buildings)

    if dropped_empty_count:
        logger.info(
            "Dropped %s EUBUCCO rows with missing or empty geometry.",
            dropped_empty_count,
        )

    invalid = ~buildings.geometry.is_valid

    if invalid.any():
        logger.warning(
            "Repairing %s invalid EUBUCCO geometries with buffer(0).",
            int(invalid.sum()),
        )
        buildings.loc[invalid, geometry_column] = buildings.loc[
            invalid, geometry_column
        ].buffer(0)

    before_valid_filter_count = len(buildings)

    buildings = buildings[~buildings.geometry.is_empty].copy()
    buildings = buildings[buildings.geometry.is_valid].copy()

    dropped_invalid_count = before_valid_filter_count - len(buildings)

    if dropped_invalid_count:
        logger.warning(
            "Dropped %s EUBUCCO rows that remained invalid or empty after repair.",
            dropped_invalid_count,
        )

    return buildings


def _dissolve_building_parts_by_id(
    buildings: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """
    Dissolve EUBUCCO building parts into single building records.

    EUBUCCO IDs may contain part suffixes such as ``abc-1`` and ``abc-2``.
    This function strips trailing ``-<number>`` suffixes, counts how many parts
    each final building has, warns when attributes disagree across parts, and
    dissolves geometries by the normalized ID.
    """
    id_col = str(EubuccoField.id)
    part_count_col = str(BuildingField.PART_COUNT)

    if id_col not in buildings.columns:
        raise KeyError(f"Cannot dissolve by id. Column {id_col!r} was not found.")

    buildings = buildings.copy()

    buildings[id_col] = (
        buildings[id_col].astype(str).str.replace(r"-\d+$", "", regex=True)
    )

    buildings[part_count_col] = buildings.groupby(id_col)[id_col].transform("size")

    _warn_on_conflicting_part_attributes(
        buildings,
        id_col=id_col,
        columns=[
            str(EubuccoField.region_id),
            str(EubuccoField.city_id),
            str(EubuccoField.type),
            str(EubuccoField.subtype),
            str(EubuccoField.height),
            str(EubuccoField.floors),
            str(EubuccoField.construction_year),
        ],
    )

    geometry_col = buildings.geometry.name

    aggfunc = {
        str(EubuccoField.region_id): _first_non_null,
        str(EubuccoField.city_id): _first_non_null,
        str(EubuccoField.type): _first_non_null,
        str(EubuccoField.subtype): _first_non_null,
        str(EubuccoField.height): _mean_or_first,
        str(EubuccoField.floors): _mean_or_first,
        str(EubuccoField.construction_year): _first_non_null,
        part_count_col: "max",
    }

    aggfunc = {
        column: func
        for column, func in aggfunc.items()
        if column in buildings.columns and column not in {id_col, geometry_col}
    }

    dissolved = buildings.dissolve(
        by=id_col,
        as_index=False,
        aggfunc=aggfunc,
    )

    logger.info(
        "Dissolved %s EUBUCCO rows into %s building records.",
        len(buildings),
        len(dissolved),
    )

    return dissolved


def _first_non_null(values: pd.Series):
    values = values.dropna()

    if values.empty:
        return pd.NA

    return values.iloc[0]


def _mean_or_first(values: pd.Series):
    non_null = values.dropna()

    if non_null.empty:
        return pd.NA

    numeric = pd.to_numeric(non_null, errors="coerce")

    if numeric.notna().all():
        return float(numeric.mean())

    return non_null.iloc[0]


def _warn_on_conflicting_part_attributes(
    buildings: gpd.GeoDataFrame,
    *,
    id_col: str,
    columns: list[str],
    max_examples: int = 5,
) -> None:
    grouped = buildings.groupby(id_col, dropna=False)

    for column in columns:
        if column not in buildings.columns:
            continue

        nunique = grouped[column].nunique(dropna=True)
        conflicting_ids = nunique[nunique > 1]

        if conflicting_ids.empty:
            continue

        examples = conflicting_ids.head(max_examples).index.tolist()

        logger.warning(
            "Column %r has conflicting values across building parts for %s "
            "dissolved building IDs. Using configured aggregation. "
            "Example IDs: %s",
            column,
            len(conflicting_ids),
            examples,
        )


def _filter_to_aoi(
    buildings: gpd.GeoDataFrame,
    *,
    aoi: BaseGeometry | gpd.GeoSeries | gpd.GeoDataFrame | str | Path,
    aoi_crs: str | None,
    target_crs: str,
) -> gpd.GeoDataFrame:
    aoi_gdf = _coerce_aoi_to_geodataframe(aoi, aoi_crs=aoi_crs)
    aoi_gdf = aoi_gdf.to_crs(target_crs)

    aoi_geometry = _union_geometries(aoi_gdf)

    before_count = len(buildings)

    buildings = buildings[buildings.geometry.intersects(aoi_geometry)].copy()

    dropped_count = before_count - len(buildings)

    if dropped_count:
        logger.info(
            "Dropped %s EUBUCCO rows outside AOI.",
            dropped_count,
        )

    return buildings


def _coerce_aoi_to_geodataframe(
    aoi: BaseGeometry | gpd.GeoSeries | gpd.GeoDataFrame | str | Path,
    *,
    aoi_crs: str | None,
) -> gpd.GeoDataFrame:
    if isinstance(aoi, gpd.GeoDataFrame):
        if aoi.crs is None and aoi_crs is None:
            raise ValueError("AOI GeoDataFrame has no CRS. Pass aoi_crs.")

        return aoi if aoi.crs is not None else aoi.set_crs(aoi_crs)

    if isinstance(aoi, gpd.GeoSeries):
        if aoi.crs is None and aoi_crs is None:
            raise ValueError("AOI GeoSeries has no CRS. Pass aoi_crs.")

        return gpd.GeoDataFrame(
            geometry=aoi,
            crs=aoi.crs or aoi_crs,
        )

    if isinstance(aoi, BaseGeometry):
        if aoi_crs is None:
            raise ValueError("AOI shapely geometry requires aoi_crs.")

        return gpd.GeoDataFrame(
            geometry=[aoi],
            crs=aoi_crs,
        )

    path = Path(aoi)
    aoi_gdf = gpd.read_file(path)

    if aoi_gdf.crs is None and aoi_crs is not None:
        aoi_gdf = aoi_gdf.set_crs(aoi_crs)

    if aoi_gdf.crs is None:
        raise ValueError(f"AOI file has no CRS: {path}")

    return aoi_gdf


def _union_geometries(gdf: gpd.GeoDataFrame) -> BaseGeometry:
    """
    Return a single geometry representing all AOI geometries.

    GeoPandas/Shapely versions differ between ``union_all`` and ``unary_union``.
    This helper supports both.
    """
    geometry = gdf.geometry

    if hasattr(geometry, "union_all"):
        return geometry.union_all()

    return geometry.unary_union


def _filter_by_area(
    buildings: gpd.GeoDataFrame,
    *,
    min_area_m2: float | None = None,
    max_area_m2: float | None = None,
) -> gpd.GeoDataFrame:
    buildings = buildings.copy()
    area_column = str(BuildingField.AREA)
    buildings[area_column] = buildings.geometry.area

    if min_area_m2 is not None:
        buildings = buildings[buildings[area_column] >= min_area_m2]

    if max_area_m2 is not None:
        buildings = buildings[buildings[area_column] <= max_area_m2]

    return buildings.reset_index(drop=True)
