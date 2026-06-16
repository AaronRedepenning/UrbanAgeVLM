from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from urban_vlm.utils import load_yaml


class PrepareInputsConfig(BaseModel):
    matched_buildings_file: Path = Path("data/interim/matched_buildings.parquet")


class PrepareOutputsConfig(BaseModel):
    all_jsonl: Path = Path("data/processed/all.jsonl")
    train_jsonl: Path | None = Path("data/processed/train.jsonl")
    val_jsonl: Path | None = Path("data/processed/val.jsonl")
    test_jsonl: Path | None = Path("data/processed/test.jsonl")
    summary_file: Path | None = Path("data/processed/summary.json")


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
    mode: Literal["percent", "fixed", "adaptive"] = "adaptive"

    # Used by mode="percent" and mode="adaptive"
    padding_ratio: float = 0.5
    min_padding_px: int = 16
    max_padding_px: int | None = 256

    # Used by mode="fixed"
    fixed_size_px: int | None = None

    # Used by mode="adaptive"
    min_size_px: int | None = 320
    max_size_px: int | None = 800
    adaptive_scale: float = 2.0

    # General behavior
    square: bool = True
    drop_edge_crops: bool = False

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

    @field_validator("fixed_size_px")
    @classmethod
    def validate_fixed_size_px(cls, value: int | None) -> int | None:
        if value is not None and value <= 0:
            raise ValueError("crop.fixed_size_px must be positive when provided.")
        return value

    @field_validator("min_size_px", "max_size_px")
    @classmethod
    def validate_size_px(cls, value: int | None) -> int | None:
        if value is not None and value <= 0:
            raise ValueError("crop min/max sizes must be positive when provided.")
        return value

    @field_validator("adaptive_scale")
    @classmethod
    def validate_adaptive_scale(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("crop.adaptive_scale must be positive.")
        return value

    @model_validator(mode="after")
    def validate_crop_mode(self) -> "CropConfig":
        if self.mode == "fixed" and self.fixed_size_px is None:
            raise ValueError("crop.fixed_size_px is required when crop.mode='fixed'.")

        if (
            self.min_size_px is not None
            and self.max_size_px is not None
            and self.min_size_px > self.max_size_px
        ):
            raise ValueError("crop.min_size_px must be <= crop.max_size_px.")

        return self


class CropImageOutputConfig(BaseModel):
    enabled: bool = False
    output_dir: Path = Path("data/processed/crops")
    image_format: Literal["png", "jpg", "jpeg", "webp"] = "png"
    overwrite: bool = False
    relative_to: Path | None = None


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
    crop_images: CropImageOutputConfig = Field(default_factory=CropImageOutputConfig)
    split: SplitConfig = Field(default_factory=SplitConfig)


def load_prepare_config(path: str | Path) -> PrepareDataConfig:
    raw: dict[str, Any] = load_yaml(path)
    return PrepareDataConfig.model_validate(raw)
