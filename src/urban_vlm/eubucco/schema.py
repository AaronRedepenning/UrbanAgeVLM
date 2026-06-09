from dataclasses import dataclass
from typing import Iterable

import geopandas as gpd
import pandas as pd


@dataclass(frozen=True)
class EubuccoField:
    # Identifiers
    id: str = "id"
    region_id: str = "region_id"
    city_id: str = "city_id"

    # Attributes
    type: str = "type"
    subtype: str = "subtype"
    height: str = "height"
    floors: str = "floors"
    construction_year: str = "construction_year"

    # Geometry
    geometry: str = "geometry"

    # Confidence
    type_confidence: str = "type_confidence"
    subtype_confidence: str = "subtype_confidence"

    height_confidence_lower: str = "height_confidence_lower"
    height_confidence_upper: str = "height_confidence_upper"

    floors_confidence_lower: str = "floors_confidence_lower"
    floors_confidence_upper: str = "floors_confidence_upper"

    construction_year_confidence_lower: str = "construction_year_confidence_lower"
    construction_year_confidence_upper: str = "construction_year_confidence_upper"

    # Sources
    geometry_source: str = "geometry_source"
    type_source: str = "type_source"
    subtype_source: str = "subtype_source"
    height_source: str = "height_source"
    floors_source: str = "floors_source"
    construction_year_source: str = "construction_year_source"

    # Source IDs
    geometry_source_id: str = "geometry_source_id"
    type_source_ids: str = "type_source_ids"
    subtype_source_ids: str = "subtype_source_ids"
    height_source_ids: str = "height_source_ids"
    floors_source_ids: str = "floors_source_ids"
    construction_year_source_ids: str = "construction_year_source_ids"

    # Source values
    subtype_raw: str = "subtype_raw"


FIELDS = EubuccoField()


IDENTIFIER_FIELDS = {
    FIELDS.id,
    FIELDS.region_id,
    FIELDS.city_id,
}

ATTRIBUTE_FIELDS = {
    FIELDS.type,
    FIELDS.subtype,
    FIELDS.height,
    FIELDS.floors,
    FIELDS.construction_year,
}

GEOMETRY_FIELDS = {
    FIELDS.geometry,
}

CONFIDENCE_FIELDS = {
    FIELDS.type_confidence,
    FIELDS.subtype_confidence,
    FIELDS.height_confidence_lower,
    FIELDS.height_confidence_upper,
    FIELDS.floors_confidence_lower,
    FIELDS.floors_confidence_upper,
    FIELDS.construction_year_confidence_lower,
    FIELDS.construction_year_confidence_upper,
}

SOURCE_FIELDS = {
    FIELDS.geometry_source,
    FIELDS.type_source,
    FIELDS.subtype_source,
    FIELDS.height_source,
    FIELDS.floors_source,
    FIELDS.construction_year_source,
}

SOURCE_ID_FIELDS = {
    FIELDS.geometry_source_id,
    FIELDS.type_source_ids,
    FIELDS.subtype_source_ids,
    FIELDS.height_source_ids,
    FIELDS.floors_source_ids,
    FIELDS.construction_year_source_ids,
}

SOURCE_VALUE_FIELDS = {
    FIELDS.subtype_raw,
}

ALL_FIELDS = (
    IDENTIFIER_FIELDS
    | ATTRIBUTE_FIELDS
    | GEOMETRY_FIELDS
    | CONFIDENCE_FIELDS
    | SOURCE_FIELDS
    | SOURCE_ID_FIELDS
    | SOURCE_VALUE_FIELDS
)

ALL_REQUIRED_FIELDS = [
    FIELDS.id,
    FIELDS.region_id,
    FIELDS.city_id,
    FIELDS.type,
    FIELDS.subtype,
    FIELDS.height,
    FIELDS.floors,
    FIELDS.construction_year,
    FIELDS.geometry,
]

NUMERIC_FIELDS = {
    FIELDS.height,
    FIELDS.floors,
    FIELDS.construction_year,
    FIELDS.type_confidence,
    FIELDS.subtype_confidence,
    FIELDS.height_confidence_lower,
    FIELDS.height_confidence_upper,
    FIELDS.floors_confidence_lower,
    FIELDS.floors_confidence_upper,
    FIELDS.construction_year_confidence_lower,
    FIELDS.construction_year_confidence_upper,
}

INTEGER_FIELDS = {
    FIELDS.construction_year,
    FIELDS.construction_year_confidence_lower,
    FIELDS.construction_year_confidence_upper,
}

CATEGORICAL_FIELDS = {
    FIELDS.type,
    FIELDS.subtype,
    FIELDS.geometry_source,
    FIELDS.type_source,
    FIELDS.subtype_source,
    FIELDS.height_source,
    FIELDS.floors_source,
    FIELDS.construction_year_source,
}


def missing_columns(df: gpd.GeoDataFrame, columns: Iterable[str]) -> set[str]:
    return set(columns) - set(df.columns)


def require_columns(df: gpd.GeoDataFrame, columns: Iterable[str]) -> None:
    missing = missing_columns(df, columns)

    if missing:
        raise ValueError(f"Missing required EUBUCCO columns: {sorted(missing)}")


def has_construction_year(df: gpd.GeoDataFrame) -> bool:
    return FIELDS.construction_year in df.columns


def coerce_numeric_columns(df: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    df = df.copy()

    for column in NUMERIC_FIELDS:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    for column in INTEGER_FIELDS:
        if column in df.columns:
            # Keep nullable integer dtype so missing values are allowed.
            df[column] = df[column].astype("Int64")

    return df
