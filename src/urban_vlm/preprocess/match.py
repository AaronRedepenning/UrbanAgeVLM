from enum import StrEnum

import geopandas as gpd

from urban_vlm.dataset.schema import BuildingField


class MatchStrategy(StrEnum):
    CENTROID_WITHIN_TILE = "centroid_within_tile"
    INTERSECTS_TILE = "intersects_tile"
    LARGEST_OVERLAP = "largest_overlap"


def match_buildings_to_tiles(
    buildings: gpd.GeoDataFrame,
    tiles: gpd.GeoDataFrame,
    *,
    strategy: MatchStrategy | str = MatchStrategy.CENTROID_WITHIN_TILE,
    keep_unmatched: bool = False,
    target_crs: str | None = None,
) -> gpd.GeoDataFrame:
    """
    Match each building to an imagery tile.
    """
    strategy = MatchStrategy(strategy)

    buildings = buildings.copy()
    tiles = tiles.copy()

    if target_crs is not None:
        buildings = buildings.to_crs(target_crs)
        tiles = tiles.to_crs(target_crs)
    else:
        if buildings.crs is None:
            raise ValueError("Buildings GeoDataFrame has no CRS.")
        if tiles.crs is None:
            raise ValueError("Tiles GeoDataFrame has no CRS.")
        if buildings.crs != tiles.crs:
            tiles = tiles.to_crs(buildings.crs)

    if strategy == MatchStrategy.CENTROID_WITHIN_TILE:
        matched = _match_centroid_within_tile(
            buildings,
            tiles,
            keep_unmatched=keep_unmatched,
        )
    elif strategy == MatchStrategy.INTERSECTS_TILE:
        matched = _match_intersects_tile(
            buildings,
            tiles,
            keep_unmatched=keep_unmatched,
        )
    elif strategy == MatchStrategy.LARGEST_OVERLAP:
        matched = _match_largest_overlap(
            buildings,
            tiles,
            keep_unmatched=keep_unmatched,
        )
    else:
        raise ValueError(f"Unsupported match strategy: {strategy}")

    return matched


def _match_centroid_within_tile(
    buildings: gpd.GeoDataFrame,
    tiles: gpd.GeoDataFrame,
    *,
    keep_unmatched: bool,
) -> gpd.GeoDataFrame:
    original_geometry_col = buildings.geometry.name
    temp_geometry_col = "__building_geometry__"
    centroid_geometry_col = "__centroid_geometry__"

    buildings_for_join = buildings.copy()

    if temp_geometry_col in buildings_for_join.columns:
        raise ValueError(f"Temporary column already exists: {temp_geometry_col}")

    if centroid_geometry_col in buildings_for_join.columns:
        raise ValueError(f"Temporary column already exists: {centroid_geometry_col}")

    buildings_for_join[temp_geometry_col] = buildings_for_join.geometry
    buildings_for_join[centroid_geometry_col] = buildings_for_join.geometry.centroid
    buildings_for_join = buildings_for_join.set_geometry(centroid_geometry_col)

    tile_cols = [
        str(BuildingField.TILE_ID),
        str(BuildingField.TILE_PATH),
        tiles.geometry.name,
    ]

    joined = gpd.sjoin(
        buildings_for_join,
        tiles[tile_cols],
        how="left" if keep_unmatched else "inner",
        predicate="within",
    )

    joined = joined.drop(columns=["index_right"], errors="ignore")
    joined = joined.drop(columns=[centroid_geometry_col], errors="ignore")

    joined = joined.set_geometry(temp_geometry_col)

    if (
        original_geometry_col in joined.columns
        and original_geometry_col != temp_geometry_col
    ):
        joined = joined.drop(columns=[original_geometry_col], errors="ignore")

    joined = joined.rename_geometry(original_geometry_col)

    return joined


def _match_intersects_tile(
    buildings: gpd.GeoDataFrame,
    tiles: gpd.GeoDataFrame,
    *,
    keep_unmatched: bool,
) -> gpd.GeoDataFrame:
    tile_cols = [
        str(BuildingField.TILE_ID),
        str(BuildingField.TILE_PATH),
        tiles.geometry.name,
    ]

    joined = gpd.sjoin(
        buildings,
        tiles[tile_cols],
        how="left" if keep_unmatched else "inner",
        predicate="intersects",
    )

    joined = joined.drop(columns=["index_right"], errors="ignore")

    return joined


def _match_largest_overlap(
    buildings: gpd.GeoDataFrame,
    tiles: gpd.GeoDataFrame,
    *,
    keep_unmatched: bool,
) -> gpd.GeoDataFrame:
    building_geometry_col = buildings.geometry.name

    buildings_with_index = buildings.copy()
    buildings_with_index["_building_row_id"] = range(len(buildings_with_index))

    tile_cols = [
        str(BuildingField.TILE_ID),
        str(BuildingField.TILE_PATH),
        tiles.geometry.name,
    ]

    candidates = gpd.sjoin(
        buildings_with_index,
        tiles[tile_cols],
        how="left",
        predicate="intersects",
    )

    candidates = candidates.drop(columns=["index_right"], errors="ignore")

    if candidates.empty:
        if keep_unmatched:
            buildings = buildings.copy()
            buildings[str(BuildingField.TILE_ID)] = None
            buildings[str(BuildingField.TILE_PATH)] = None
            return buildings
        return candidates

    tile_geom_lookup = tiles.set_index(str(BuildingField.TILE_ID))[tiles.geometry.name]

    candidates["_overlap_area"] = candidates.apply(
        lambda row: (
            row.geometry.intersection(
                tile_geom_lookup.loc[row[str(BuildingField.TILE_ID)]]
            ).area
            if row[str(BuildingField.TILE_ID)] in tile_geom_lookup.index
            else 0
        ),
        axis=1,
    )

    candidates = candidates.sort_values(
        ["_building_row_id", "_overlap_area"],
        ascending=[True, False],
    )

    best = candidates.drop_duplicates("_building_row_id", keep="first")

    if not keep_unmatched:
        best = best[best[str(BuildingField.TILE_ID)].notna()].copy()

    best = best.drop(columns=["_building_row_id", "_overlap_area"], errors="ignore")
    best = best.set_geometry(building_geometry_col)

    return best
