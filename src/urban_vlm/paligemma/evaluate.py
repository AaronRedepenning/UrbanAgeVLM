import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, median
from typing import Any

from rich.align import Align
from rich.console import Console
from rich.table import Table

from urban_vlm.paligemma.config import PaliGemmaTask

UNKNOWN_VALUES = {
    "unknown",
    "unk",
    "none",
    "null",
    "n/a",
    "na",
    "",
}

YEAR_PATTERN = re.compile(
    r"(?<!\d)(\d{4})(?:\s*(?:'s|s))?(?!\d)",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class PredictionMetricRecord:
    id: str | None
    task: str
    target_raw: Any
    prediction_raw: Any
    target: int | None
    prediction: int | None
    error: int | None
    absolute_error: int | None
    status: str
    error_bucket: str | None


@dataclass(frozen=True)
class PredictionEvaluationSummary:
    task: str
    input_jsonl: Path
    total_records: int
    evaluated_records: int
    missing_targets: int
    unknown_targets: int
    invalid_targets: int
    missing_predictions: int
    unknown_predictions: int
    invalid_predictions: int
    exact_accuracy: float | None
    tolerance_accuracy: dict[str, float]
    mae: float | None
    mse: float | None
    rmse: float | None
    median_absolute_error: float | None
    mean_error: float | None
    error_buckets: dict[str, int]
    per_target: dict[str, dict[str, Any]]
    metrics_json: Path | None = None
    per_record_metrics_jsonl: Path | None = None

    @property
    def prediction_coverage(self) -> float | None:
        valid_targets = (
            self.total_records
            - self.missing_targets
            - self.unknown_targets
            - self.invalid_targets
        )

        if valid_targets == 0:
            return None

        return self.evaluated_records / valid_targets

    def to_json_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["input_jsonl"] = str(self.input_jsonl)

        if self.metrics_json is not None:
            data["metrics_json"] = str(self.metrics_json)

        if self.per_record_metrics_jsonl is not None:
            data["per_record_metrics_jsonl"] = str(self.per_record_metrics_jsonl)

        data["prediction_coverage"] = self.prediction_coverage
        return data


def evaluate_prediction_jsonl(
    input_jsonl: str | Path,
    *,
    task: PaliGemmaTask | str | None = None,
    metrics_json: str | Path | None = None,
    per_record_metrics_jsonl: str | Path | None = None,
    tolerances: list[int] | None = None,
    min_year: int = 1800,
    max_year: int = 2035,
    show_progress: bool = True,
) -> PredictionEvaluationSummary:
    input_jsonl = Path(input_jsonl)
    records = _read_jsonl(input_jsonl)

    if task is None:
        task = _infer_task(records)

    task = PaliGemmaTask(task)

    if tolerances is None:
        if task == PaliGemmaTask.BUILDING_DECADE:
            tolerances = [0, 10, 20, 30]
        else:
            tolerances = [0, 5, 10, 20, 30]

    tolerances = sorted(set(tolerances))

    metric_records: list[PredictionMetricRecord] = []

    missing_targets = 0
    unknown_targets = 0
    invalid_targets = 0
    missing_predictions = 0
    unknown_predictions = 0
    invalid_predictions = 0

    errors: list[int] = []
    absolute_errors: list[int] = []
    error_buckets: Counter[str] = Counter()

    for record in records:
        target_raw = record.get("target")
        prediction_raw = record.get("prediction")

        target, target_status = parse_temporal_value(
            target_raw,
            task=task,
            min_year=min_year,
            max_year=max_year,
        )
        prediction, prediction_status = parse_temporal_value(
            prediction_raw,
            task=task,
            min_year=min_year,
            max_year=max_year,
        )

        if target_status == "missing":
            missing_targets += 1
        elif target_status == "unknown":
            unknown_targets += 1
        elif target_status == "invalid":
            invalid_targets += 1

        if prediction_status == "missing":
            missing_predictions += 1
        elif prediction_status == "unknown":
            unknown_predictions += 1
        elif prediction_status == "invalid":
            invalid_predictions += 1

        if target is None:
            metric_record = PredictionMetricRecord(
                id=record.get("id"),
                task=task.value,
                target_raw=target_raw,
                prediction_raw=prediction_raw,
                target=None,
                prediction=prediction,
                error=None,
                absolute_error=None,
                status=f"{target_status}_target",
                error_bucket=None,
            )
            metric_records.append(metric_record)
            continue

        if prediction is None:
            metric_record = PredictionMetricRecord(
                id=record.get("id"),
                task=task.value,
                target_raw=target_raw,
                prediction_raw=prediction_raw,
                target=target,
                prediction=None,
                error=None,
                absolute_error=None,
                status=f"{prediction_status}_prediction",
                error_bucket=None,
            )
            metric_records.append(metric_record)
            continue

        error = prediction - target
        absolute_error = abs(error)
        bucket = make_error_bucket(absolute_error, task=task)

        errors.append(error)
        absolute_errors.append(absolute_error)
        error_buckets[bucket] += 1

        metric_records.append(
            PredictionMetricRecord(
                id=record.get("id"),
                task=task.value,
                target_raw=target_raw,
                prediction_raw=prediction_raw,
                target=target,
                prediction=prediction,
                error=error,
                absolute_error=absolute_error,
                status="evaluated",
                error_bucket=bucket,
            )
        )

    evaluated_records = len(errors)

    tolerance_accuracy = {
        f"within_{tolerance}": _safe_accuracy(
            sum(error <= tolerance for error in absolute_errors),
            evaluated_records,
        )
        for tolerance in tolerances
    }

    mse = _safe_mean([error**2 for error in errors])

    summary = PredictionEvaluationSummary(
        task=task.value,
        input_jsonl=input_jsonl,
        total_records=len(records),
        evaluated_records=evaluated_records,
        missing_targets=missing_targets,
        unknown_targets=unknown_targets,
        invalid_targets=invalid_targets,
        missing_predictions=missing_predictions,
        unknown_predictions=unknown_predictions,
        invalid_predictions=invalid_predictions,
        exact_accuracy=tolerance_accuracy.get("within_0"),
        tolerance_accuracy=tolerance_accuracy,
        mae=_safe_mean(absolute_errors),
        mse=mse,
        rmse=None if mse is None else math.sqrt(mse),
        median_absolute_error=(
            None if not absolute_errors else float(median(absolute_errors))
        ),
        mean_error=_safe_mean(errors),
        error_buckets=dict(error_buckets),
        per_target=summarize_per_target(metric_records, tolerances=tolerances),
        metrics_json=Path(metrics_json) if metrics_json is not None else None,
        per_record_metrics_jsonl=(
            Path(per_record_metrics_jsonl)
            if per_record_metrics_jsonl is not None
            else None
        ),
    )

    if metrics_json is not None:
        _write_json(summary.to_json_dict(), Path(metrics_json))

    if per_record_metrics_jsonl is not None:
        _write_jsonl(
            [asdict(record) for record in metric_records],
            Path(per_record_metrics_jsonl),
        )

    if show_progress:
        _print_summary(summary)

    return summary


def parse_temporal_value(
    value: Any,
    *,
    task: PaliGemmaTask | str,
    min_year: int = 1800,
    max_year: int = 2035,
    range_policy: str = "first",
) -> tuple[int | None, str | None]:
    task = PaliGemmaTask(task)

    if value is None:
        return None, "missing"

    text = str(value).strip().lower()

    if text in UNKNOWN_VALUES:
        return None, "unknown"

    years = [int(match.group(1)) for match in YEAR_PATTERN.finditer(text)]

    years = [year for year in years if min_year <= year <= max_year]

    if not years:
        return None, "invalid"

    year = _select_year_from_prediction(
        years,
        task=task,
        range_policy=range_policy,
    )

    if task == PaliGemmaTask.BUILDING_DECADE:
        year = year // 10 * 10

    return year, None


def _select_year_from_prediction(
    years: list[int],
    *,
    task: PaliGemmaTask,
    range_policy: str,
) -> int:
    if len(years) == 1:
        return years[0]

    if range_policy == "first":
        return years[0]

    if range_policy == "midpoint":
        return round(sum(years) / len(years))

    raise ValueError(f"Unsupported range policy: {range_policy}")


def make_error_bucket(
    absolute_error: int,
    *,
    task: PaliGemmaTask | str,
) -> str:
    task = PaliGemmaTask(task)

    if task == PaliGemmaTask.BUILDING_DECADE:
        decades = absolute_error // 10

        if decades == 0:
            return "exact"

        if decades == 1:
            return "off_by_1_decade"

        if decades == 2:
            return "off_by_2_decades"

        if decades == 3:
            return "off_by_3_decades"

        return "off_by_4plus_decades"

    if absolute_error == 0:
        return "exact"

    if absolute_error <= 5:
        return "off_by_1_to_5_years"

    if absolute_error <= 10:
        return "off_by_6_to_10_years"

    if absolute_error <= 20:
        return "off_by_11_to_20_years"

    if absolute_error <= 30:
        return "off_by_21_to_30_years"

    return "off_by_31plus_years"


def summarize_per_target(
    records: list[PredictionMetricRecord],
    *,
    tolerances: list[int],
) -> dict[str, dict[str, Any]]:
    grouped: dict[int, list[PredictionMetricRecord]] = defaultdict(list)

    for record in records:
        if record.status == "evaluated" and record.target is not None:
            grouped[record.target].append(record)

    summaries: dict[str, dict[str, Any]] = {}

    for target, target_records in sorted(grouped.items()):
        absolute_errors = [
            record.absolute_error
            for record in target_records
            if record.absolute_error is not None
        ]
        errors = [record.error for record in target_records if record.error is not None]

        n = len(target_records)

        summaries[str(target)] = {
            "n": n,
            "mae": _safe_mean(absolute_errors),
            "rmse": _safe_rmse(errors),
            "mean_error": _safe_mean(errors),
            "tolerance_accuracy": {
                f"within_{tolerance}": _safe_accuracy(
                    sum(error <= tolerance for error in absolute_errors),
                    n,
                )
                for tolerance in tolerances
            },
        }

    return summaries


def _infer_task(records: list[dict[str, Any]]) -> PaliGemmaTask:
    for record in records:
        task = record.get("task")
        if task:
            return PaliGemmaTask(task)

    raise ValueError("Could not infer task because no record has a `task` field.")


def _safe_mean(values: list[int] | list[float]) -> float | None:
    if not values:
        return None

    return float(mean(values))


def _safe_rmse(errors: list[int]) -> float | None:
    if not errors:
        return None

    return math.sqrt(mean([error**2 for error in errors]))


def _safe_accuracy(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None

    return numerator / denominator


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _write_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _write_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


def _print_summary(summary: PredictionEvaluationSummary) -> None:
    console = Console()

    table = Table(
        title="Prediction evaluation summary",
        show_header=True,
        header_style="bold",
    )

    table.add_column("Metric")
    table.add_column("Value", justify="right")

    table.add_row("Task", summary.task)
    table.add_row("Total records", f"{summary.total_records:,}")
    table.add_row("Evaluated records", f"{summary.evaluated_records:,}")
    table.add_row(
        "Prediction coverage",
        (
            "—"
            if summary.prediction_coverage is None
            else f"{summary.prediction_coverage:.1%}"
        ),
    )
    table.add_row(
        "Exact accuracy",
        "—" if summary.exact_accuracy is None else f"{summary.exact_accuracy:.1%}",
    )

    for key, value in summary.tolerance_accuracy.items():
        table.add_row(
            key.replace("_", " "),
            "—" if value is None else f"{value:.1%}",
        )

    table.add_section()
    table.add_row("MAE", "—" if summary.mae is None else f"{summary.mae:.2f}")
    table.add_row("MSE", "—" if summary.mse is None else f"{summary.mse:.2f}")
    table.add_row("RMSE", "—" if summary.rmse is None else f"{summary.rmse:.2f}")
    table.add_row(
        "Median absolute error",
        (
            "—"
            if summary.median_absolute_error is None
            else f"{summary.median_absolute_error:.2f}"
        ),
    )
    table.add_row(
        "Mean error / bias",
        "—" if summary.mean_error is None else f"{summary.mean_error:.2f}",
    )

    table.add_section()
    table.add_row("Missing predictions", f"{summary.missing_predictions:,}")
    table.add_row("Unknown predictions", f"{summary.unknown_predictions:,}")
    table.add_row("Invalid predictions", f"{summary.invalid_predictions:,}")

    console.print()
    console.print(Align.center(table))
