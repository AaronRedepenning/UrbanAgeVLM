from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from urban_vlm.utils import load_yaml


class PaliGemmaTask(StrEnum):
    BUILDING_YEAR = "building_year"
    BUILDING_DECADE = "building_decade"


TorchDtypeName = Literal["auto", "float32", "float16", "bfloat16"]


class PaliGemmaModelConfig(BaseModel):
    model_id: str = "google/paligemma2-3b-mix-448"
    torch_dtype: TorchDtypeName = "bfloat16"
    device_map: str | None = "auto"


class PaliGemmaDataConfig(BaseModel):
    train_jsonl: Path | None = None
    val_jsonl: Path | None = None
    test_jsonl: Path | None = None
    predict_jsonl: Path | None = None
    image_root: Path | None = None
    max_records: int | None = None

    @field_validator("max_records")
    @classmethod
    def validate_max_records(cls, value: int | None) -> int | None:
        if value is not None and value <= 0:
            raise ValueError("max_records must be positive when provided.")
        return value


class PaliGemmaGenerationConfig(BaseModel):
    max_new_tokens: int = 32
    do_sample: bool = False
    temperature: float | None = None
    top_p: float | None = None
    num_beams: int = 1


class PaliGemmaLoraConfig(BaseModel):
    enabled: bool = False
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: list[str] = Field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )


class PaliGemmaTrainingConfig(BaseModel):
    output_dir: Path = Path("outputs/checkpoints/paligemma")
    num_train_epochs: float = 1.0
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-5
    weight_decay: float = 0.0
    warmup_ratio: float = 0.03
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500
    save_total_limit: int = 2
    bf16: bool = True
    fp16: bool = False
    gradient_checkpointing: bool = True
    remove_unused_columns: bool = False
    report_to: list[str] = Field(default_factory=list)
    seed: int = 42


class PaliGemmaConfig(BaseModel):
    base_dir: Path | None = None
    task: PaliGemmaTask = PaliGemmaTask.BUILDING_DECADE
    model: PaliGemmaModelConfig = Field(default_factory=PaliGemmaModelConfig)
    data: PaliGemmaDataConfig = Field(default_factory=PaliGemmaDataConfig)
    generation: PaliGemmaGenerationConfig = Field(
        default_factory=PaliGemmaGenerationConfig
    )
    lora: PaliGemmaLoraConfig = Field(default_factory=PaliGemmaLoraConfig)
    training: PaliGemmaTrainingConfig = Field(default_factory=PaliGemmaTrainingConfig)

    @model_validator(mode="after")
    def validate_base_dir(self) -> "PaliGemmaConfig":
        if self.base_dir is not None:
            self._resolve_paths(self.base_dir)
        return self

    def _resolve_paths(self, base_dir: Path) -> None:
        for path_attr in (
            "train_jsonl",
            "val_jsonl",
            "test_jsonl",
            "predict_jsonl",
            "image_root",
        ):
            path_value = getattr(self.data, path_attr)
            if path_value is not None and not path_value.is_absolute():
                setattr(self.data, path_attr, base_dir / path_value)


def load_paligemma_config(path: str | Path) -> PaliGemmaConfig:
    return PaliGemmaConfig.model_validate(load_yaml(path))
