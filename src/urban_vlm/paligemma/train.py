import os
from pathlib import Path

from transformers import Trainer, TrainingArguments

from urban_vlm.paligemma.config import PaliGemmaConfig
from urban_vlm.paligemma.data import JsonlDataset, PaliGemmaCollator
from urban_vlm.paligemma.model import (
    apply_lora_if_enabled,
    load_paligemma_model,
    load_paligemma_processor,
)


def train_paligemma(cfg: PaliGemmaConfig) -> Trainer:
    if cfg.data.train_jsonl is None:
        raise ValueError("cfg.data.train_jsonl is required for training.")

    output_dir = Path(cfg.training.output_dir)
    final_dir = output_dir / "final"

    train_dataset = JsonlDataset(
        cfg.data.train_jsonl,
        max_records=cfg.data.max_records,
    )

    eval_dataset = None
    has_eval = cfg.data.val_jsonl is not None

    if has_eval:
        eval_dataset = JsonlDataset(
            cfg.data.val_jsonl,
            max_records=cfg.data.max_records,
        )

    processor = load_paligemma_processor(cfg.model)

    model = load_paligemma_model(cfg.model)
    model = apply_lora_if_enabled(model, cfg.lora)

    if cfg.training.gradient_checkpointing:
        model.config.use_cache = False

    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()

    collator = PaliGemmaCollator(
        processor=processor,
        task=cfg.task,
        image_root=cfg.data.image_root,
        train=True,
        return_metadata=False,
    )

    # Wandb config
    os.environ["WANDB_PROJECT"] = "paligemma-building-age"
    os.environ["WANDB_LOG_MODEL"] = "false"
    os.environ["WANDB_WATCH"] = "false"

    args = _training_arguments(
        cfg,
        has_eval=has_eval,
        run_name=_make_run_name(cfg),
    )

    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        args=args,
        processing_class=processor,
    )

    trainer.train()

    trainer.save_model(final_dir)
    processor.save_pretrained(final_dir)

    return trainer


def _training_arguments(
    cfg: PaliGemmaConfig,
    *,
    has_eval: bool,
    run_name: str | None = None,
) -> TrainingArguments:
    eval_strategy = "steps" if has_eval else "no"

    load_best_model_at_end = has_eval

    if has_eval and cfg.training.save_steps % cfg.training.eval_steps != 0:
        raise ValueError(
            "When load_best_model_at_end=True with step-based evaluation, "
            "save_steps must be a multiple of eval_steps."
        )

    return TrainingArguments(
        output_dir=str(cfg.training.output_dir),
        run_name=run_name,
        num_train_epochs=cfg.training.num_train_epochs,
        remove_unused_columns=cfg.training.remove_unused_columns,
        per_device_train_batch_size=cfg.training.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.training.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        warmup_ratio=cfg.training.warmup_ratio,
        learning_rate=cfg.training.learning_rate,
        lr_scheduler_type="cosine",
        weight_decay=cfg.training.weight_decay,
        max_grad_norm=1.0,
        logging_steps=cfg.training.logging_steps,
        optim="adamw_torch",
        save_strategy="steps",
        save_steps=cfg.training.save_steps,
        save_total_limit=cfg.training.save_total_limit,
        fp16=cfg.training.fp16,
        bf16=cfg.training.bf16,
        gradient_checkpointing=cfg.training.gradient_checkpointing,
        eval_strategy=eval_strategy,
        eval_steps=cfg.training.eval_steps if has_eval else None,
        load_best_model_at_end=load_best_model_at_end,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=cfg.training.report_to,
        seed=cfg.training.seed,
        data_seed=cfg.training.seed,
    )


def _make_run_name(cfg: PaliGemmaConfig):
    dataset_id = Path(cfg.base_dir).name
    model_id = Path(cfg.model.model_id).name

    return (
        f"{cfg.task.name}"
        f"__{model_id}"
        f"__lr{cfg.training.learning_rate:g}"
        f"__{dataset_id}"
        f"__s{cfg.training.seed}"
    )
