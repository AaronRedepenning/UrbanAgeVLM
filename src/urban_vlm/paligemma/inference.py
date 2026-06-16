from pathlib import Path
from typing import Any

import torch
from rich.progress import track
from torch.utils.data import DataLoader

from urban_vlm.dataset.jsonl import write_jsonl
from urban_vlm.paligemma.config import PaliGemmaConfig
from urban_vlm.paligemma.data import JsonlDataset, PaliGemmaCollator, build_target
from urban_vlm.paligemma.model import load_paligemma_model, load_paligemma_processor


class PaliGemmaPredictor:
    def __init__(self, cfg: PaliGemmaConfig) -> None:
        self.cfg = cfg
        self.processor = load_paligemma_processor(cfg.model)
        self.model = load_paligemma_model(cfg.model)
        self.model.eval()

    @torch.inference_mode()
    def predict_jsonl(
        self,
        input_jsonl: str | Path,
        *,
        output_jsonl: str | Path | None = None,
        batch_size: int | None = None,
        show_progress: bool = True,
    ) -> list[dict[str, Any]]:
        dataset = JsonlDataset(
            input_jsonl,
            max_records=self.cfg.data.max_records,
        )

        collator = PaliGemmaCollator(
            processor=self.processor,
            task=self.cfg.task,
            image_root=self.cfg.data.image_root,
            train=False,
            return_metadata=True,
        )

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size or self.cfg.training.per_device_eval_batch_size,
            shuffle=False,
            collate_fn=collator,
        )

        predictions: list[dict[str, Any]] = []

        iterator = (
            track(dataloader, description="Running PaliGemma inference")
            if show_progress
            else dataloader
        )

        for batch in iterator:
            model_inputs = batch["model_inputs"]
            model_inputs = model_inputs.to(self.model.device)

            output_ids = self.model.generate(
                **model_inputs,
                **self._generation_kwargs(),
            )

            generated_ids = output_ids[:, model_inputs["input_ids"].shape[-1] :]

            texts = self.processor.batch_decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

            for record, prompt, text in zip(
                batch["records"],
                batch["prompts"],
                texts,
                strict=True,
            ):
                predictions.append(
                    {
                        "id": record.get("id"),
                        "task": str(self.cfg.task),
                        "prompt": prompt,
                        "target": build_target(record, self.cfg.task),
                        "prediction": text.strip(),
                    }
                )

        if output_jsonl is not None:
            write_jsonl(predictions, output_jsonl)

        return predictions

    def _generation_kwargs(self) -> dict[str, Any]:
        cfg = self.cfg.generation

        kwargs: dict[str, Any] = {
            "max_new_tokens": cfg.max_new_tokens,
            "do_sample": cfg.do_sample,
            "num_beams": cfg.num_beams,
        }

        if cfg.temperature is not None:
            kwargs["temperature"] = cfg.temperature

        if cfg.top_p is not None:
            kwargs["top_p"] = cfg.top_p

        return kwargs


def predict_jsonl(
    cfg: PaliGemmaConfig,
    *,
    input_jsonl: Path | None = None,
    output_jsonl: Path | None = None,
    batch_size: int | None = None,
    show_progress: bool = True,
) -> list[dict[str, Any]]:
    input_path = input_jsonl or cfg.data.predict_jsonl or cfg.data.test_jsonl

    if input_path is None:
        raise ValueError("No prediction JSONL provided.")

    predictor = PaliGemmaPredictor(cfg)

    return predictor.predict_jsonl(
        input_path,
        output_jsonl=output_jsonl,
        batch_size=batch_size,
        show_progress=show_progress,
    )
