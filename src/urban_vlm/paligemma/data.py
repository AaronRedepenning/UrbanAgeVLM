from dataclasses import dataclass
from pathlib import Path
from typing import Any

from torch.utils.data import Dataset

from urban_vlm.dataset.jsonl import read_jsonl
from urban_vlm.paligemma.config import PaliGemmaTask
from urban_vlm.paligemma.images import load_training_image_from_record
from urban_vlm.paligemma.prompts import build_prompt, build_target


class JsonlDataset(Dataset):
    def __init__(
        self,
        jsonl_path: str | Path,
        *,
        max_records: int | None = None,
    ) -> None:
        records = read_jsonl(jsonl_path)
        if max_records is not None:
            records = records[:max_records]

        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]

        return {
            "id": record.get("id"),
            "record": record,
        }


@dataclass
class PaliGemmaCollator:
    processor: Any
    task: PaliGemmaTask
    image_root: Path | None = None
    train: bool = True
    return_metadata: bool = False

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        records = [feature["record"] for feature in features]
        ids = [feature.get("id") for feature in features]

        images = [
            load_training_image_from_record(record, image_root=self.image_root)
            for record in records
        ]

        prompts = [build_prompt(record, self.task) for record in records]

        if self.train:
            targets = [build_target(record, self.task) for record in records]

            model_inputs = self.processor(
                images=images,
                text=prompts,
                suffix=targets,
                return_tensors="pt",
                padding=True,
            )
        else:
            model_inputs = self.processor(
                images=images,
                text=prompts,
                return_tensors="pt",
                padding=True,
            )

        if not self.return_metadata:
            return model_inputs

        return {
            "model_inputs": model_inputs,
            "ids": ids,
            "records": records,
            "prompts": prompts,
        }
