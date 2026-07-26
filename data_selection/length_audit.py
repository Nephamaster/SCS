"""Audit prompt/response token-length distributions for a JSON/JSONL dataset.

The default accounting is content-based and additive:

    total_tokens = prompt_tokens + response_tokens
    assistant_ratio = response_tokens / total_tokens

Prompt tokens are tokens from non-assistant messages; response tokens are
tokens from assistant messages. Chat-template control tokens are not included,
so the report is directly comparable across datasets when the same tokenizer is
used.

Example:
    python data_selection/length_audit.py \
        --input data/sft/random/random_01.jsonl \
        --tokenizer-model /share/project/wuhaiming/data/models/Llama-3.1-8B
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Iterator

from tqdm.auto import tqdm

ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_TOKENIZER_MODEL = "/share/project/wuhaiming/data/models/Llama-3.1-8B"

if __package__ in {None, ""}:  # pragma: no cover - direct script execution
    sys.path.insert(0, str(ROOT_DIR))

from src.data.normalize import normalize_messages


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT_DIR / path


def iter_json_records(path: Path) -> Iterator[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
                if not isinstance(record, dict):
                    raise ValueError(f"Expected an object at {path}:{line_number}")
                yield record
        return

    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as handle:
            records = json.load(handle)
        if not isinstance(records, list):
            raise ValueError(f"JSON dataset must contain an array: {path}")
        for index, record in enumerate(records):
            if not isinstance(record, dict):
                raise ValueError(f"Expected an object at {path}[{index}]")
            yield record
        return

    raise ValueError(f"Only .jsonl and .json inputs are supported: {path}")


def percentile(values: list[float], fraction: float) -> float:
    """Linear-interpolated percentile, matching common statistical tools."""
    if not values:
        raise ValueError("Cannot calculate a percentile for an empty list")
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def summarize_lengths(values: list[int]) -> dict[str, float | int]:
    if not values:
        raise ValueError("No valid records were found")
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": round(statistics.mean(values), 4),
        "P50": round(percentile(values, 0.50), 4),
        "P90": round(percentile(values, 0.90), 4),
        "P95": round(percentile(values, 0.95), 4),
        "P99": round(percentile(values, 0.99), 4),
    }


def load_tokenizer(model_name_or_path: str):
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "Length auditing requires transformers. Install the project requirements."
        ) from exc

    model_path = Path(model_name_or_path).expanduser()
    return AutoTokenizer.from_pretrained(
        str(model_path),
        use_fast=True,
        local_files_only=model_path.exists(),
    )


def count_content_tokens(tokenizer: Any, messages: list[dict[str, str]]) -> tuple[int, int]:
    prompt_tokens = 0
    response_tokens = 0
    for message in messages:
        token_count = len(
            tokenizer.encode(message["content"], add_special_tokens=False)
        )
        if message["role"] == "assistant":
            response_tokens += token_count
        else:
            prompt_tokens += token_count
    return prompt_tokens, response_tokens


def audit_dataset(input_path: Path, tokenizer: Any) -> dict[str, Any]:
    prompt_lengths: list[int] = []
    response_lengths: list[int] = []
    total_lengths: list[int] = []
    assistant_ratios: list[float] = []
    skipped_records = 0

    for record_index, record in enumerate(
        tqdm(
            iter_json_records(input_path),
            desc="Audit token lengths",
            unit="row",
        )
    ):
        try:
            messages = normalize_messages(record, record_index=record_index)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid record {record_index}: {exc}") from exc

        prompt_tokens, response_tokens = count_content_tokens(tokenizer, messages)
        total_tokens = prompt_tokens + response_tokens
        if total_tokens == 0:
            skipped_records += 1
            continue

        prompt_lengths.append(prompt_tokens)
        response_lengths.append(response_tokens)
        total_lengths.append(total_tokens)
        assistant_ratios.append(response_tokens / total_tokens)

    if not total_lengths:
        raise ValueError("No non-empty records were available for length auditing")

    total_token_count = sum(total_lengths)
    response_token_count = sum(response_lengths)
    return {
        "input": str(input_path),
        "record_count": len(total_lengths),
        "skipped_empty_records": skipped_records,
        "prompt_tokens": summarize_lengths(prompt_lengths),
        "response_tokens": summarize_lengths(response_lengths),
        "total_tokens": summarize_lengths(total_lengths),
        "effective_assistant_token_ratio": {
            "corpus_weighted": round(response_token_count / total_token_count, 6),
            "mean_per_sample": round(statistics.mean(assistant_ratios), 6),
            "P50": round(percentile(assistant_ratios, 0.50), 6),
            "P90": round(percentile(assistant_ratios, 0.90), 6),
            "P95": round(percentile(assistant_ratios, 0.95), 6),
            "P99": round(percentile(assistant_ratios, 0.99), 6),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit prompt, response, total token lengths and assistant-token ratios."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Dataset JSONL or JSON array to audit.",
    )
    parser.add_argument(
        "--tokenizer-model",
        default=DEFAULT_TOKENIZER_MODEL,
        help=f"Tokenizer model/path (default: {DEFAULT_TOKENIZER_MODEL}).",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path for saving the JSON report.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = resolve_path(args.input)
    tokenizer = load_tokenizer(args.tokenizer_model)
    report = audit_dataset(input_path, tokenizer)
    report["tokenizer_model"] = args.tokenizer_model

    if args.output_json is not None:
        output_path = resolve_path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
