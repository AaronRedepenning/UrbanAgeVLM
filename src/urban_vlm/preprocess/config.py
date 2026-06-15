from pathlib import Path

from pydantic import BaseModel, Field, field_validator

from urban_vlm.preprocess.match import MatchStrategy
from urban_vlm.utils import load_yaml


class EubuccoPreprocessConfig(BaseModel):
    input_dir: Path = Path("data/raw/eubucco")
    file_glob: str = "**/*.parquet"
    dissolve_by_id: bool = False
    require_construction_year: bool = True
    min_area_m2: float | None = 20.0
    max_area_m2: float | None = None

    @field_validator("min_area_m2", "max_area_m2")
    @classmethod
    def validate_area(cls, value: float | None) -> float | None:
        if value is not None and value <= 0:
            raise ValueError("Area filters must be positive when provided.")
        return value


class TilesPreprocessConfig(BaseModel):
    input_dir: Path = Path("data/raw/bayern/dop20")
    file_glob: str = "**/*.tif"


class MatchPreprocessConfig(BaseModel):
    strategy: MatchStrategy = MatchStrategy.CENTROID_WITHIN_TILE
    keep_unmatched: bool = False
    target_crs: str | None = None


class PreprocessOutputsConfig(BaseModel):
    cleaned_buildings_file: Path = Path("data/interim/buildings_clean.parquet")
    tile_index_file: Path = Path("data/interim/tile_index.parquet")
    matched_buildings_file: Path = Path("data/interim/matched_buildings.parquet")


class PreprocessConfig(BaseModel):
    eubucco: EubuccoPreprocessConfig = Field(default_factory=EubuccoPreprocessConfig)
    tiles: TilesPreprocessConfig = Field(default_factory=TilesPreprocessConfig)
    match: MatchPreprocessConfig = Field(default_factory=MatchPreprocessConfig)
    outputs: PreprocessOutputsConfig = Field(default_factory=PreprocessOutputsConfig)


def load_preprocess_config(path: str | Path) -> PreprocessConfig:
    return PreprocessConfig.model_validate(load_yaml(path))
