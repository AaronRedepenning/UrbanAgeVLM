import json
import re
from pathlib import Path

import torch
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)
from torch.utils.data import DataLoader


def collate_fn(batch: list[dict]) -> dict:
    return {
        "ids": [item["id"] for item in batch],
        "images": [item["image"] for item in batch],
        "prefixes": [item["prefix"] for item in batch],
        "suffixes": [item["suffix"] for item in batch],
        "records": [item["record"] for item in batch],
    }


@torch.inference_mode()
def run_inference(
    *,
    model,
    processor,
    dataset,
    output_jsonl: str | Path,
    batch_size: int = 1,
    max_new_tokens: int = 8,
    do_sample: bool = False,
) -> int:
    output_jsonl = Path(output_jsonl)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    total = len(loader)
    count = 0

    progress = Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
    )

    with progress:
        task_id = progress.add_task("Running PaliGemma inference", total=total)

        with output_jsonl.open("w", encoding="utf-8") as f:
            for batch in loader:
                inputs = processor(
                    text=batch["prefixes"],
                    images=batch["images"],
                    return_tensors="pt",
                    padding=True,
                ).to(model.device)

                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    num_beams=1,
                )

                input_token_len = inputs["input_ids"].shape[1]
                new_token_ids = generated_ids[:, input_token_len:]

                decoded = processor.batch_decode(
                    new_token_ids,
                    skip_special_tokens=True,
                )

                for record_id, target, raw_text in zip(
                    batch["ids"],
                    batch["suffixes"],
                    decoded,
                ):
                    pred_decade = parse_decade(raw_text)
                    target_decade = parse_decade(target)

                    out = {
                        "id": record_id,
                        "target": target,
                        "target_decade": target_decade,
                        "prediction_raw": raw_text,
                        "prediction_decade": pred_decade,
                    }

                    f.write(json.dumps(out, ensure_ascii=False) + "\n")
                    count += 1
                    progress.update(task_id, completed=count)

    return count


def parse_year(text: str) -> int | None:
    match = re.search(r"\b(18|19|20)\d{2}\b", text)

    if match is None:
        return None

    return int(match.group(0))


def parse_decade(text: str) -> int | None:
    decade_match = re.search(r"\b((18|19|20)\d0)s?\b", text)

    if decade_match is not None:
        return int(decade_match.group(1))

    year = parse_year(text)

    if year is not None:
        return year // 10 * 10

    return None
