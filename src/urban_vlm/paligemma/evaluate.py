from urban_vlm.paligemma.data import JsonlDataset, TargetTask
from urban_vlm.paligemma.inference import run_inference
from urban_vlm.paligemma.model import load_paligemma


def evaluate_paligemma() -> None:
    dataset = JsonlDataset(
        "data/processed/single_buildings_test.jsonl",
        task=TargetTask.CONSTRUCTION_DECADE,
        max_examples=100,
    )

    processor, model = load_paligemma()

    run_inference(
        model=model,
        processor=processor,
        dataset=dataset,
        output_jsonl="outputs/paligemma/baseline_predictions.jsonl",
        max_new_tokens=16,
    )
