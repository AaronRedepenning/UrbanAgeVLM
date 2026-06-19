from typing import Any

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import PaliGemmaForConditionalGeneration, PaliGemmaProcessor

from urban_vlm.paligemma.config import PaliGemmaLoraConfig, PaliGemmaModelConfig


def load_paligemma_processor(cfg: PaliGemmaModelConfig):
    return PaliGemmaProcessor.from_pretrained(
        cfg.model_id,
    )


def load_paligemma_model(cfg: PaliGemmaModelConfig):
    kwargs: dict[str, Any] = {}

    torch_dtype = _torch_dtype(cfg.torch_dtype)

    if torch_dtype is not None:
        kwargs["torch_dtype"] = torch_dtype

    if cfg.device_map is not None:
        kwargs["device_map"] = cfg.device_map

    model = PaliGemmaForConditionalGeneration.from_pretrained(
        cfg.model_id,
        **kwargs,
    )

    # If device_map is None, Transformers loads the model on CPU.
    # Move it to a single best device for simple notebooks/scripts.
    if cfg.device_map is None:
        model = model.to(_default_device())

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


def _default_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")
