from pathlib import Path

from peft import LoraConfig, get_peft_model
from transformers import Trainer, TrainingArguments

from urban_vlm.paligemma.data import JsonlDataset, TargetTask
from urban_vlm.paligemma.model import load_paligemma


class PaliGemmaTrainCollator:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, batch: list[dict]) -> dict:
        images = [item["image"] for item in batch]
        prefixes = [item["prefix"] for item in batch]
        suffixes = [item["suffix"] for item in batch]

        inputs = self.processor(
            text=prefixes,
            images=images,
            suffix=suffixes,
            return_tensors="pt",
            padding=True,
        )

        return inputs


def train_paligemma(
    train_path: str | Path = "data/processed/single_buildings_train.jsonl",
    val_path: str | Path = "data/processed/single_buildings_val.jsonl",
    out_dir: str | Path = "outputs/paligemma/checkpoints/decade_lora",
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = JsonlDataset(
        train_path,
        task=TargetTask.CONSTRUCTION_DECADE,
        max_examples=10,
    )
    val_dataset = JsonlDataset(
        val_path,
        task=TargetTask.CONSTRUCTION_DECADE,
        max_examples=10,
    )

    processor, model = load_paligemma(train=True)

    lora_config = LoraConfig(
        r=32,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=[
            "q_proj",
            "o_proj",
            "k_proj",
            "v_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            "multi_modal_projector.linear",
        ],
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    args = TrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=1,
        remove_unused_columns=False,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        warmup_ratio=0.05,
        learning_rate=2e-4,
        weight_decay=0.01,
        logging_steps=25,
        optim="paged_adamw_8bit",
        save_strategy="steps",
        save_steps=250,
        save_total_limit=2,
        fp16=True,
        bf16=False,
        eval_strategy="steps",
        eval_steps=250,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        run_name="paligemma_decade_lora",
        seed=42,
        data_seed=42,
    )

    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=PaliGemmaTrainCollator(processor),
        args=args,
        processing_class=processor,
    )

    trainer.train()

    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)
