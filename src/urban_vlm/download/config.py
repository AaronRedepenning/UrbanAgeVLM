from pathlib import Path

from pydantic import BaseModel, Field, HttpUrl, field_validator

from urban_vlm.utils import load_yaml


class NutsDownloadConfig(BaseModel):
    enabled: bool = True
    out_dir: Path = Path("data/raw/nuts")
    url: HttpUrl


class EubuccoDownloadConfig(BaseModel):
    enabled: bool = True
    out_dir: Path = Path("data/raw/eubucco")
    version: str = "v0.2"
    nuts_ids: list[str] = Field(default_factory=list)

    @field_validator("nuts_ids")
    @classmethod
    def normalize_nuts_ids(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("eubucco.nuts_ids must contain at least one NUTS ID")
        return [item.strip().upper() for item in value]


class BayernDownloadConfig(BaseModel):
    enabled: bool = True
    out_dir: Path = Path("data/raw/bayern/dop20")
    meta4_urls: list[HttpUrl] = Field(default_factory=list)

    @field_validator("meta4_urls")
    @classmethod
    def validate_meta4_urls(cls, value: list[HttpUrl]) -> list[HttpUrl]:
        if not value:
            raise ValueError("bayern.meta4_urls must contain at least one URL")
        return value


class BerlinDownloadConfig(BaseModel):
    enabled: bool = True
    out_dir: Path = Path("data/raw/berlin/dop_2025_fruehjahr")
    atom_url: HttpUrl


class DownloadOptionsConfig(BaseModel):
    overwrite: bool = False
    timeout_seconds: int = 60
    max_workers: int = Field(default=8, ge=1)
    show_progress: bool = True


class DownloadConfig(BaseModel):
    nuts: NutsDownloadConfig
    eubucco: EubuccoDownloadConfig
    bayern: BayernDownloadConfig
    berlin: BerlinDownloadConfig
    download: DownloadOptionsConfig = Field(default_factory=DownloadOptionsConfig)


def load_download_config(path: str | Path) -> DownloadConfig:
    return DownloadConfig.model_validate(load_yaml(path))
