from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from urban_vlm.utils import load_yaml


class PrepareInputsConfig(BaseModel):
    matched_buildings_file: Path = Path(
        "data/interim/matches/matched_buildings.parquet"
    )


class PrepareOutputsConfig(BaseModel):
    all_jsonl: Path = Path("data/processed/single_buildings_all.jsonl")
    train_jsonl: Path | None = Path("data/processed/single_buildings_train.jsonl")
    val_jsonl: Path | None = Path("data/processed/single_buildings_val.jsonl")
    test_jsonl: Path | None = Path("data/processed/single_buildings_test.jsonl")
    summary_file: Path | None = Path(
        "data/processed/prepare_training_data_summary.json"
    )


class RecordsConfig(BaseModel):
    max_records: int | None = None
    shuffle: bool = True
    seed: int = 42

    @field_validator("max_records")
    @classmethod
    def validate_max_records(cls, value: int | None) -> int | None:
        if value is not None and value <= 0:
            raise ValueError("records.max_records must be positive when provided.")
        return value


class CropConfig(BaseModel):
    padding_ratio: float = 0.5
    min_padding_px: int = 16
    max_padding_px: int | None = 256
    edge_policy: Literal["keep_clipped", "drop_clipped"] = "keep_clipped"

    @field_validator("padding_ratio")
    @classmethod
    def validate_padding_ratio(cls, value: float) -> float:
        if value < 0:
            raise ValueError("crop.padding_ratio must be non-negative.")
        return value

    @field_validator("min_padding_px")
    @classmethod
    def validate_min_padding_px(cls, value: int) -> int:
        if value < 0:
            raise ValueError("crop.min_padding_px must be non-negative.")
        return value

    @field_validator("max_padding_px")
    @classmethod
    def validate_max_padding_px(cls, value: int | None) -> int | None:
        if value is not None and value < 0:
            raise ValueError("crop.max_padding_px must be non-negative when provided.")
        return value


class SplitConfig(BaseModel):
    enabled: bool = True
    group_key: str = "image"
    train: float = 0.8
    val: float = 0.1
    test: float = 0.1
    seed: int = 42

    @model_validator(mode="after")
    def validate_split_fractions(self) -> "SplitConfig":
        total = self.train + self.val + self.test

        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"Split fractions must sum to 1.0. "
                f"Got train + val + test = {total}."
            )

        if min(self.train, self.val, self.test) < 0:
            raise ValueError("Split fractions must be non-negative.")

        return self


class PrepareDataConfig(BaseModel):
    inputs: PrepareInputsConfig
    outputs: PrepareOutputsConfig = Field(default_factory=PrepareOutputsConfig)
    records: RecordsConfig = Field(default_factory=RecordsConfig)
    crop: CropConfig = Field(default_factory=CropConfig)
    split: SplitConfig = Field(default_factory=SplitConfig)


def load_prepare_config(path: str | Path) -> PrepareDataConfig:
    raw: dict[str, Any] = load_yaml(path)
    return PrepareDataConfig.model_validate(raw)
