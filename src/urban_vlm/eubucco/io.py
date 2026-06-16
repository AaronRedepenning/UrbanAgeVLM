import warnings
from dataclasses import dataclass
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


@dataclass(frozen=True)
class EubuccoReadStats:
    path: Path
    raw_rows: int
    after_basic_clean: int
    after_aoi: int
    after_dissolve: int
    after_construction_year: int
    final_rows: int
    dropped_missing_or_empty_geometry: int = 0
    repaired_invalid_geometry: int = 0
    dropped_invalid_geometry: int = 0
    dropped_outside_aoi: int = 0
    dissolved_rows_removed: int = 0
    dropped_missing_construction_year: int = 0
    dropped_below_min_area: int = 0
    dropped_above_max_area: int = 0

    @property
    def total_dropped(self) -> int:
        return self.raw_rows - self.final_rows


@dataclass(frozen=True)
class EubuccoReadResult:
    buildings: gpd.GeoDataFrame
    stats: EubuccoReadStats


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
) -> EubuccoReadResult:
    path = Path(path)

    buildings = read_geodataframe(
        path,
        columns=ALL_REQUIRED_FIELDS,
    )
    require_columns(buildings, ALL_REQUIRED_FIELDS)

    raw_rows = len(buildings)

    buildings = buildings.to_crs(target_crs)

    (
        buildings,
        dropped_missing_or_empty_geometry,
        repaired_invalid_geometry,
        dropped_invalid_geometry,
    ) = _clean_basic_eubucco(buildings)

    after_basic_clean = len(buildings)

    dropped_outside_aoi = 0
    if aoi is not None:
        before = len(buildings)
        buildings = _filter_to_aoi(
            buildings,
            aoi=aoi,
            aoi_crs=aoi_crs,
            target_crs=target_crs,
        )
        dropped_outside_aoi = before - len(buildings)

    after_aoi = len(buildings)

    dissolved_rows_removed = 0
    if dissolve_by_id:
        before = len(buildings)
        buildings = _dissolve_building_parts_by_id(buildings)
        dissolved_rows_removed = before - len(buildings)
    else:
        buildings = buildings.copy()
        buildings[str(BuildingField.PART_COUNT)] = 1

    after_dissolve = len(buildings)

    dropped_missing_construction_year = 0
    if require_construction_year:
        construction_year_col = str(EubuccoField.construction_year)
        before = len(buildings)
        buildings = buildings[buildings[construction_year_col].notna()].copy()
        dropped_missing_construction_year = before - len(buildings)

    after_construction_year = len(buildings)

    (
        buildings,
        dropped_below_min_area,
        dropped_above_max_area,
    ) = _filter_by_area(
        buildings,
        min_area_m2=min_area_m2,
        max_area_m2=max_area_m2,
    )

    stats = EubuccoReadStats(
        path=path,
        raw_rows=raw_rows,
        after_basic_clean=after_basic_clean,
        after_aoi=after_aoi,
        after_dissolve=after_dissolve,
        after_construction_year=after_construction_year,
        final_rows=len(buildings),
        dropped_missing_or_empty_geometry=dropped_missing_or_empty_geometry,
        repaired_invalid_geometry=repaired_invalid_geometry,
        dropped_invalid_geometry=dropped_invalid_geometry,
        dropped_outside_aoi=dropped_outside_aoi,
        dissolved_rows_removed=dissolved_rows_removed,
        dropped_missing_construction_year=dropped_missing_construction_year,
        dropped_below_min_area=dropped_below_min_area,
        dropped_above_max_area=dropped_above_max_area,
    )

    return EubuccoReadResult(buildings=buildings, stats=stats)


def _clean_basic_eubucco(
    buildings: gpd.GeoDataFrame,
) -> tuple[gpd.GeoDataFrame, int, int, int]:
    buildings = buildings.copy()
    buildings = coerce_numeric_columns(buildings)

    geometry_col = buildings.geometry.name

    before = len(buildings)
    missing_or_empty = buildings[geometry_col].isna() | buildings.geometry.is_empty
    buildings = buildings[~missing_or_empty].copy()
    dropped_missing_or_empty_geometry = before - len(buildings)

    invalid = ~buildings.geometry.is_valid
    repaired_invalid_geometry = int(invalid.sum())

    if repaired_invalid_geometry:
        buildings.loc[invalid, geometry_col] = buildings.loc[
            invalid,
            geometry_col,
        ].buffer(0)

    before_valid_filter = len(buildings)

    buildings = buildings[~buildings.geometry.is_empty].copy()
    buildings = buildings[buildings.geometry.is_valid].copy()

    dropped_invalid_geometry = before_valid_filter - len(buildings)

    return (
        buildings,
        dropped_missing_or_empty_geometry,
        repaired_invalid_geometry,
        dropped_invalid_geometry,
    )


def _dissolve_building_parts_by_id(
    buildings: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
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

    return buildings.dissolve(
        by=id_col,
        as_index=False,
        aggfunc=aggfunc,
    )


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

        warnings.warn(
            f"Column {column!r} has conflicting values across building parts "
            f"for {len(conflicting_ids)} dissolved building IDs. "
            f"Using configured aggregation. Example IDs: {examples}",
            stacklevel=2,
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

    return buildings[buildings.geometry.intersects(aoi_geometry)].copy()


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
    geometry = gdf.geometry

    if hasattr(geometry, "union_all"):
        return geometry.union_all()

    return geometry.unary_union


def _filter_by_area(
    buildings: gpd.GeoDataFrame,
    *,
    min_area_m2: float | None = None,
    max_area_m2: float | None = None,
) -> tuple[gpd.GeoDataFrame, int, int]:
    buildings = buildings.copy()

    area_col = str(BuildingField.AREA)
    buildings[area_col] = buildings.geometry.area

    dropped_below_min_area = 0
    dropped_above_max_area = 0

    if min_area_m2 is not None:
        before = len(buildings)
        buildings = buildings[buildings[area_col] >= min_area_m2].copy()
        dropped_below_min_area = before - len(buildings)

    if max_area_m2 is not None:
        before = len(buildings)
        buildings = buildings[buildings[area_col] <= max_area_m2].copy()
        dropped_above_max_area = before - len(buildings)

    return (
        buildings.reset_index(drop=True),
        dropped_below_min_area,
        dropped_above_max_area,
    )
