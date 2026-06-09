from enum import StrEnum


class BuildingField(StrEnum):
    PART_COUNT = "part_count"
    AREA_M2 = "area_m2"

    TILE_ID = "tile_id"
    TILE_PATH = "tile_path"

    CROP_PATH = "crop_path"
    CROP_BOX_PX = "crop_box_px"
