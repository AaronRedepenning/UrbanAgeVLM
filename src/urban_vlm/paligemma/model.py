import torch
from transformers import (
    PaliGemmaForConditionalGeneration,
    PaliGemmaProcessor,
)


def load_paligemma(model_name: str = "google/paligemma2-3b-mix-448"):
    processor = PaliGemmaProcessor.from_pretrained(model_name)

    model = PaliGemmaForConditionalGeneration.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model.eval()
    # model.config.use_cache = True

    return processor, model
