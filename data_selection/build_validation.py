"""Sample a validation set from the original Tulu and COIG-CQIA data.

The frozen 300K candidate pool is used only as an exact exclusion set.  Source
rows are adapted with the same adapters as candidate-pool construction, then
sampled uniformly with reservoir sampling.  No MinHash or token-count
calculation is performed.

Example:
    python data_selection/build_validation.py \
        --tulu-path data/raw/tulu-3-sft-mixture \
        --coig-path data/raw/COIG-CQIA \
        --candidate-path data/candidate/v1/candidate_messages.jsonl \
        --output data/dev/candidate_1000.jsonl \
        --num-samples 1000 \
        --seed 42
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any, Iterable, Iterator

from tqdm.auto import tqdm

ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_TULU_PATH = ROOT_DIR / "data" / "raw" / "tulu-3-sft-mixture"
DEFAULT_COIG_PATH = ROOT_DIR / "data" / "raw" / "COIG-CQIA"
DEFAULT_CANDIDATE_PATH = (
    ROOT_DIR / "data" / "candidate" / "v1" / "candidate_messages.jsonl"
)
DEFAULT_OUTPUT_PATH = ROOT_DIR / "data" / "dev" / "candidate_1000.jsonl"

TULU_RELATIVE_FILES = tuple(
    Path("data") / f"train-{index:05d}-of-00006.parquet"
    for index in range(6)
)
COIG_RELATIVE_FILES = (Path("COIG-CQIA-full.jsonl"),)

if __package__ in {None, ""}:  # pragma: no cover - direct script execution
    sys.path.insert(0, str(ROOT_DIR))

from src.data.coig_cqia_adapter import adapt_coig_row
from src.data.loaders import iter_records
from src.data.normalize import normalize_messages, validate_messages
from src.data.tulu3_adapter import adapt_tulu_row


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT_DIR / path


def iter_json_records(path: Path) -> Iterator[dict[str, Any]]:
    """Stream JSONL or a JSON array."""
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


def normalized_messages(record: dict[str, Any], record_index: int) -> list[dict[str, str]]:
    return normalize_messages(record, record_index=record_index)


def message_key(messages: list[dict[str, str]]) -> str:
    payload = json.dumps(
        messages,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_candidate_exclusion_index(
    candidate_path: Path,
) -> tuple[set[str], set[str], int]:
    """Index the frozen 300K candidate pool by ID and exact messages."""
    candidate_ids: set[str] = set()
    candidate_keys: set[str] = set()
    row_count = 0
    for row_count, record in enumerate(
        tqdm(
            iter_json_records(candidate_path),
            desc="Index 300K candidate pool",
            unit="row",
        ),
        start=1,
    ):
        sample_id = record.get("sample_id")
        if isinstance(sample_id, str) and sample_id:
            candidate_ids.add(sample_id)
        candidate_keys.add(message_key(normalized_messages(record, row_count - 1)))
    return candidate_ids, candidate_keys, row_count


def iter_original_messages(
    tulu_path: Path,
    coig_path: Path,
) -> Iterator[dict[str, Any]]:
    """Yield normalized valid records from the original source datasets."""
    tulu_rows = iter_records(
        tulu_path,
        relative_paths=TULU_RELATIVE_FILES,
        split_name="train",
        description="Load Tulu train shards",
    )
    for row_index, row in enumerate(
        tqdm(tulu_rows, desc="Normalize original Tulu", unit="row")
    ):
        try:
            record = adapt_tulu_row(row, row_index)
            messages = normalized_messages(record, row_index)
            valid, _ = validate_messages(messages)
            if valid:
                yield {"sample_id": record["sample_id"], "messages": messages}
        except (TypeError, ValueError):
            continue

    coig_rows = iter_records(
        coig_path,
        relative_paths=COIG_RELATIVE_FILES,
        description="Load COIG-CQIA full JSONL",
    )
    for row_index, row in enumerate(
        tqdm(coig_rows, desc="Normalize original COIG-CQIA", unit="row")
    ):
        try:
            record = adapt_coig_row(row, row_index)
            messages = normalized_messages(record, row_index)
            valid, _ = validate_messages(messages)
            if valid:
                yield {"sample_id": record["sample_id"], "messages": messages}
        except (TypeError, ValueError):
            continue


def reservoir_sample(
    source_records: Iterable[dict[str, Any]],
    candidate_ids: set[str],
    candidate_keys: set[str],
    num_samples: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Uniformly sample eligible source rows with bounded memory."""
    rng = random.Random(seed)
    selected: list[dict[str, Any]] = []
    seen_source_keys: set[str] = set()
    source_count = 0
    excluded_by_id = 0
    excluded_by_text = 0
    excluded_internal_duplicate = 0
    eligible_count = 0

    for source_count, record in enumerate(
        tqdm(source_records, desc="Sample validation candidates", unit="row"),
        start=1,
    ):
        sample_id = record["sample_id"]
        messages = record["messages"]
        key = message_key(messages)

        if sample_id in candidate_ids:
            excluded_by_id += 1
            continue
        if key in candidate_keys:
            excluded_by_text += 1
            continue
        if key in seen_source_keys:
            excluded_internal_duplicate += 1
            continue
        seen_source_keys.add(key)

        eligible_count += 1
        output_record = {"messages": messages}
        if len(selected) < num_samples:
            selected.append(output_record)
        else:
            replacement_index = rng.randrange(eligible_count)
            if replacement_index < num_samples:
                selected[replacement_index] = output_record

    stats = {
        "source_count": source_count,
        "eligible_count": eligible_count,
        "excluded_by_candidate_sample_id": excluded_by_id,
        "excluded_by_candidate_messages": excluded_by_text,
        "excluded_internal_source_duplicates": excluded_internal_duplicate,
    }
    return selected, stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sample an exact-deduplicated validation JSONL from original "
            "Tulu and COIG-CQIA data."
        )
    )
    parser.add_argument("--tulu-path", type=Path, default=DEFAULT_TULU_PATH)
    parser.add_argument("--coig-path", type=Path, default=DEFAULT_COIG_PATH)
    parser.add_argument(
        "--candidate-path",
        type=Path,
        default=DEFAULT_CANDIDATE_PATH,
        help="Frozen 300K candidate pool used as the exclusion set.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Validation JSONL output (default: {DEFAULT_OUTPUT_PATH}).",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1000,
        help="Number of validation samples (default: 1000).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_samples <= 0:
        raise ValueError("--num-samples must be positive")

    tulu_path = resolve_path(args.tulu_path)
    coig_path = resolve_path(args.coig_path)
    candidate_path = resolve_path(args.candidate_path)
    output_path = resolve_path(args.output)

    if output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output already exists: {output_path}. Use --overwrite to replace it."
        )
    if output_path.resolve() in {
        tulu_path.resolve(),
        coig_path.resolve(),
        candidate_path.resolve(),
    }:
        raise ValueError("Output path must differ from all input paths.")

    candidate_ids, candidate_keys, candidate_count = build_candidate_exclusion_index(
        candidate_path
    )
    selected, stats = reservoir_sample(
        iter_original_messages(tulu_path, coig_path),
        candidate_ids,
        candidate_keys,
        args.num_samples,
        args.seed,
    )
    if len(selected) < args.num_samples:
        raise ValueError(
            f"Only {len(selected)} eligible source rows remain after deduplication; "
            f"cannot sample {args.num_samples}. Stats: {stats}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in tqdm(selected, desc="Write validation set", unit="row"):
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")

    print(
        json.dumps(
            {
                "candidate_count": candidate_count,
                "candidate_unique_ids": len(candidate_ids),
                "candidate_unique_messages": len(candidate_keys),
                "validation_count": len(selected),
                "output": str(output_path),
                **stats,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
