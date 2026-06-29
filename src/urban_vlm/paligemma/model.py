from typing import Any

import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import PaliGemmaForConditionalGeneration, PaliGemmaProcessor

from urban_vlm.paligemma.config import PaliGemmaLoraConfig, PaliGemmaModelConfig


def load_paligemma_processor(cfg: PaliGemmaModelConfig):
    processor_id = cfg.processor_id or cfg.model_id
    return PaliGemmaProcessor.from_pretrained(
        processor_id,
    )


def load_paligemma_model(cfg: PaliGemmaModelConfig):
    torch_dtype = _torch_dtype(cfg.torch_dtype)

    kwargs: dict[str, Any] = {
        "low_cpu_mem_usage": True,
    }

    if torch_dtype is not None:
        kwargs["torch_dtype"] = torch_dtype

    # Prefer device_map="auto" on CUDA instead of loading then model.to(cuda)
    if cfg.device_map is not None:
        kwargs["device_map"] = cfg.device_map
    elif torch.cuda.is_available():
        kwargs["device_map"] = "auto"

    base_model = PaliGemmaForConditionalGeneration.from_pretrained(
        cfg.model_id,
        **kwargs,
    )

    if cfg.adapter_id is not None:
        model = PeftModel.from_pretrained(
            base_model,
            cfg.adapter_id,
            is_trainable=False,
        )
    else:
        model = base_model

    return model


def apply_lora_if_enabled(model, cfg: PaliGemmaLoraConfig):
    if not cfg.enabled:
        return model

    peft_config = LoraConfig(
        r=cfg.r,
        lora_alpha=cfg.alpha,
        lora_dropout=cfg.dropout,
        target_modules=cfg.target_modules,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    return get_peft_model(model, peft_config)


def _torch_dtype(name: str):
    if name == "auto":
        return None

    if name == "float32":
        return torch.float32

    if name == "float16":
        return torch.float16

    if name == "bfloat16":
        return torch.bfloat16

    raise ValueError(f"Unsupported torch dtype: {name}")
