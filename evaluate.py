from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Sequence


Point = float | datetime
Interval = tuple[Point, Point]

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_GT_JSON = REPO_ROOT / "VisAnomBench" / "VisAnomBench_test.json"
FIELDNAMES = ["TP", "FP", "FN", "overlap", "precision", "recall", "f1"]

INDEX_TAG_RE = re.compile(r"<index>\s*(.*?)\s*</index>", re.IGNORECASE | re.DOTALL)
NUMBER_RE = re.compile(r"[-+]?\d*\.\d+|[-+]?\d+")


def completion_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return str(value.get("content") or value.get("text") or value.get("model_output") or "")
    if isinstance(value, list):
        parts = []
        for item in value:
            text = completion_text(item)
            if text:
                parts.append(text)
        return "\n".join(parts).strip()
    return str(value)


def normalize_path(path: Any) -> str:
    return str(path or "").replace("\\", "/")


def parse_datetime(text: str, base_date=None) -> datetime:
    text = text.strip()

    for fmt in (
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%d %I:%M:%S %p",
        "%Y-%m-%d %I:%M %p",
    ):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            pass

    try:
        return datetime.fromisoformat(text)
    except ValueError:
        pass

    if base_date is None:
        raise ValueError(f"Need a base date to parse time-only value: {text}")

    for fmt in ("%H:%M:%S", "%H:%M", "%I:%M:%S %p", "%I:%M %p"):
        try:
            parsed_time = datetime.strptime(text, fmt).time()
            return datetime.combine(base_date, parsed_time)
        except ValueError:
            pass

    fixed = re.sub(r"\s*(AM|PM)$", "", text, flags=re.IGNORECASE).strip()
    for fmt in ("%H:%M:%S", "%H:%M"):
        try:
            parsed_time = datetime.strptime(fixed, fmt).time()
            return datetime.combine(base_date, parsed_time)
        except ValueError:
            pass

    raise ValueError(f"Unrecognized datetime value: {text}")


def parse_interval(text: str, is_datetime: bool, base_date=None) -> Interval | None:
    if "," not in text:
        return None

    start_text, end_text = [part.strip() for part in text.split(",", 1)]

    try:
        if is_datetime:
            start = parse_datetime(start_text, base_date=base_date)
            end = parse_datetime(end_text, base_date=start.date())
        else:
            numbers = [float(value) for value in NUMBER_RE.findall(text)]
            if len(numbers) != 2:
                return None
            start, end = numbers
    except ValueError:
        return None

    if end < start:
        start, end = end, start
    return start, end


def unique_intervals(intervals: Sequence[Interval]) -> list[Interval]:
    seen = set()
    out = []
    for interval in intervals:
        if interval in seen:
            continue
        seen.add(interval)
        out.append(interval)
    return out


def intervals_from_string(value: str, base_date=None) -> list[Interval]:
    if not value:
        return []

    candidates = re.findall(r"\(([^()]+)\)", value)
    if not candidates:
        return []

    is_datetime = bool(re.search(r"\d{4}-\d{2}-\d{2}", candidates[0]))
    intervals = []
    for candidate in candidates:
        interval = parse_interval(candidate, is_datetime=is_datetime, base_date=base_date)
        if interval is None:
            continue
        if is_datetime and base_date is None:
            base_date = interval[0].date()
        intervals.append(interval)

    return unique_intervals(intervals)


def coerce_point(value: Any, is_datetime: bool, base_date=None) -> Point:
    if isinstance(value, datetime):
        return value
    if is_datetime:
        return parse_datetime(str(value), base_date=base_date)
    return float(value)


def gt_intervals(value: Any) -> list[Interval]:
    if isinstance(value, str):
        return intervals_from_string(value)

    if isinstance(value, dict):
        value = [value]

    if not isinstance(value, list):
        return []

    intervals = []
    for item in value:
        if isinstance(item, dict):
            start, end = item.get("start"), item.get("end")
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            start, end = item[0], item[1]
        else:
            continue

        if start is None or end is None:
            continue

        is_datetime = isinstance(start, str) and bool(re.search(r"\d{4}-\d{2}-\d{2}", start))
        try:
            start_point = coerce_point(start, is_datetime=is_datetime)
            end_point = coerce_point(
                end,
                is_datetime=is_datetime,
                base_date=start_point.date() if is_datetime else None,
            )
        except ValueError:
            continue

        if end_point < start_point:
            start_point, end_point = end_point, start_point
        intervals.append((start_point, end_point))

    return unique_intervals(intervals)


def predicted_intervals(model_output: str, ground_truth: Sequence[Interval]) -> list[Interval]:
    if not model_output:
        return []

    is_datetime = bool(ground_truth and isinstance(ground_truth[0][0], datetime))
    base_date = ground_truth[0][0].date() if is_datetime else None
    intervals = []

    tag_contents = INDEX_TAG_RE.findall(model_output)
    search_space = tag_contents or [model_output]

    for text in search_space:
        candidates = re.findall(r"\(([^()]+)\)", text)
        if not candidates and "," in text:
            candidates = [text]
        for candidate in candidates:
            interval = parse_interval(candidate, is_datetime=is_datetime, base_date=base_date)
            if interval is not None:
                intervals.append(interval)

    return unique_intervals(intervals)


def pad_interval(interval: Interval, left: int, right: int) -> Interval:
    start, end = interval
    if isinstance(start, datetime):
        return start - timedelta(minutes=left), end + timedelta(minutes=right)
    return float(start) - left, float(end) + right


def intervals_overlap(first: Interval, second: Interval) -> bool:
    return not (first[1] < second[0] or second[1] < first[0])


def interval_points(intervals: Sequence[Interval]) -> set[int]:
    points = set()
    for start, end in intervals:
        if isinstance(start, datetime):
            start_value = int(start.timestamp() // 60)
            end_value = int(end.timestamp() // 60)
        else:
            start_value = int(start)
            end_value = int(end)

        if end_value >= start_value:
            points.update(range(start_value, end_value + 1))

    return points


def overlap_score(ground_truth: Sequence[Interval], predictions: Sequence[Interval]) -> float:
    gt_points = interval_points(ground_truth)
    pred_points = interval_points(predictions)
    return len(gt_points & pred_points) / max(len(gt_points), len(pred_points), 1)


def prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def count_metrics(
    ground_truth: Sequence[Interval],
    predictions: Sequence[Interval],
    pred_pad_left: int,
    pred_pad_right: int,
) -> dict[str, float | int]:
    padded_predictions = [
        pad_interval(interval, pred_pad_left, pred_pad_right) for interval in predictions
    ]
    matched_gt = set()
    tp = 0
    fp = 0

    for prediction in padded_predictions:
        match = None
        for index, truth in enumerate(ground_truth):
            if index not in matched_gt and intervals_overlap(prediction, truth):
                match = index
                break

        if match is None:
            fp += 1
        else:
            matched_gt.add(match)
            tp += 1

    fn = len(ground_truth) - len(matched_gt)
    precision, recall, f1 = prf(tp, fp, fn)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "overlap": overlap_score(ground_truth, padded_predictions),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def gt_lookup(gt_data: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    by_image = {}
    by_basename = {}

    for record in gt_data.values():
        if not isinstance(record, dict):
            continue
        image = normalize_path(record.get("image"))
        if image:
            by_image[image] = record
            by_basename.setdefault(os.path.basename(image), record)

    return gt_data, by_image, by_basename


def model_records(model_data: Any) -> list[dict[str, Any]]:
    if isinstance(model_data, list):
        records = []
        for index, record in enumerate(model_data):
            if not isinstance(record, dict):
                continue
            records.append(
                {
                    "key": record.get("key"),
                    "id": record.get("id", index),
                    "image": normalize_path(record.get("image")),
                    "model_output": completion_text(record.get("model_output")),
                }
            )
        return records

    if isinstance(model_data, dict):
        records = []
        for key, value in model_data.items():
            if isinstance(value, dict):
                record_id = value.get("id")
                image = normalize_path(value.get("image"))
                output = completion_text(value.get("model_output", value.get("output", "")))
            else:
                record_id = None
                image = ""
                output = completion_text(value)

            records.append(
                {
                    "key": key,
                    "id": record_id,
                    "image": image,
                    "model_output": output,
                }
            )
        return records

    raise ValueError("Model JSON must be a list or object.")


def find_gt_record(
    record: dict[str, Any],
    by_key: dict[str, Any],
    by_image: dict[str, Any],
    by_basename: dict[str, Any],
) -> Any | None:
    for key in (record.get("key"), record.get("id")):
        if key is None:
            continue
        key = str(key)
        if key in by_key:
            return by_key[key]
        gutentag_key = key if key.startswith("GutenTAG_") else f"GutenTAG_{key}"
        if gutentag_key in by_key:
            return by_key[gutentag_key]

    image = normalize_path(record.get("image"))
    if image in by_image:
        return by_image[image]

    basename = os.path.basename(image)
    if basename in by_basename:
        return by_basename[basename]

    return None


def evaluate(
    model_json: Path,
    gt_json: Path = DEFAULT_GT_JSON,
    pred_pad_left: int = 100,
    pred_pad_right: int = 100,
) -> dict[str, float | int]:
    with model_json.open("r", encoding="utf-8") as f:
        model_data = json.load(f)
    with gt_json.open("r", encoding="utf-8") as f:
        gt_data = json.load(f)

    if not isinstance(gt_data, dict):
        raise ValueError(f"GT JSON must be an object keyed by sample id: {gt_json}")

    by_key, by_image, by_basename = gt_lookup(gt_data)
    totals = {"tp": 0, "fp": 0, "fn": 0}
    overlap_sum = 0.0
    matched_rows = 0
    missing_gt_rows = 0

    for record in model_records(model_data):
        truth_record = find_gt_record(record, by_key, by_image, by_basename)
        if truth_record is None:
            missing_gt_rows += 1
            continue

        truth_value = (
            truth_record.get("anomalies", "") if isinstance(truth_record, dict) else truth_record
        )
        truth_intervals = gt_intervals(truth_value)
        prediction_intervals = predicted_intervals(record["model_output"], truth_intervals)
        metrics = count_metrics(
            truth_intervals,
            prediction_intervals,
            pred_pad_left=pred_pad_left,
            pred_pad_right=pred_pad_right,
        )

        totals["tp"] += int(metrics["tp"])
        totals["fp"] += int(metrics["fp"])
        totals["fn"] += int(metrics["fn"])
        overlap_sum += float(metrics["overlap"])
        matched_rows += 1

    if missing_gt_rows:
        print(f"[WARN] skipped {missing_gt_rows} row(s) with no matching GT", file=sys.stderr)

    precision, recall, f1 = prf(totals["tp"], totals["fp"], totals["fn"])
    return {
        "TP": totals["tp"],
        "FP": totals["fp"],
        "FN": totals["fn"],
        "overlap": overlap_sum / matched_rows if matched_rows else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute TP, FP, FN, overlap, precision, recall, and F1."
    )
    parser.add_argument("model_json", type=Path, help="Model output JSON.")
    parser.add_argument("--gt-json", type=Path, default=DEFAULT_GT_JSON)
    parser.add_argument("--out-csv", type=Path)
    parser.add_argument("--pred-pad-left", type=int, default=100)
    parser.add_argument("--pred-pad-right", type=int, default=100)
    args = parser.parse_args()

    row = evaluate(
        model_json=args.model_json,
        gt_json=args.gt_json,
        pred_pad_left=args.pred_pad_left,
        pred_pad_right=args.pred_pad_right,
    )

    if args.out_csv:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.out_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            writer.writeheader()
            writer.writerow(row)
        return

    writer = csv.DictWriter(sys.stdout, fieldnames=FIELDNAMES)
    writer.writeheader()
    writer.writerow(row)


if __name__ == "__main__":
    main()
