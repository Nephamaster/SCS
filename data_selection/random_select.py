"""Generate reproducible random ms-swift subsets from the frozen candidate pool.

Each output record keeps the zero-based ``candidate_index`` of the source
record, together with ``sample_id``, ``source`` and ``messages``.  No new DOC
files, flattened documents, or token counts are generated here.  Downstream
selection methods can use ``candidate_index`` to read the corresponding row
from the full candidate-pool feature arrays.

Example:
    python data_selection/random_select.py \
        --input data/candidate/v1/candidate_messages.jsonl \
        --num_groups 12 \
        --sample_size 10000 \
        --seed 123
"""

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT_DIR / "data" / "candidate" / "v1" / "candidate_messages.jsonl"
DEFAULT_SFT_OUTPUT_DIR = ROOT_DIR / "data" / "sft" / "random"


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT_DIR / path


def _validate_candidate_record(record: Any, record_index: int) -> dict[str, Any]:
    if not isinstance(record, dict):
        raise ValueError(f"Candidate record {record_index} must be an object.")

    for field in ("sample_id", "source"):
        value = record.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"Candidate record {record_index} must contain a non-empty '{field}'."
            )

    messages = record.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError(
            f"Candidate record {record_index} must contain a non-empty 'messages' list."
        )
    for turn_index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise ValueError(
                f"Candidate record {record_index}, turn {turn_index} must be an object."
            )
        if not isinstance(message.get("role"), str) or not message["role"].strip():
            raise ValueError(
                f"Candidate record {record_index}, turn {turn_index} has an invalid role."
            )
        if not isinstance(message.get("content"), str):
            raise ValueError(
                f"Candidate record {record_index}, turn {turn_index} content must be a string."
            )

    return record


def load_candidate_records(input_path: Path) -> list[dict[str, Any]]:
    """Load and validate the frozen ``candidate_messages.jsonl`` pool."""
    if not input_path.exists():
        raise FileNotFoundError(f"Candidate pool not found: {input_path}")
    if input_path.suffix.lower() != ".jsonl":
        raise ValueError(
            "Random selection expects the frozen candidate JSONL file "
            "candidate_messages.jsonl, not the ms-swift-only candidate_sft.jsonl."
        )

    records: list[dict[str, Any]] = []
    sample_ids: set[str] = set()
    with input_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(
            tqdm(handle, desc="Load candidate pool", unit="row"), start=1
        ):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in candidate pool at line {line_number}: {exc}"
                ) from exc

            record_index = len(records)
            record = _validate_candidate_record(record, record_index)
            sample_id = record["sample_id"]
            if sample_id in sample_ids:
                raise ValueError(f"Duplicate candidate sample_id: {sample_id}")
            sample_ids.add(sample_id)
            records.append(record)

    if not records:
        raise ValueError(f"Candidate pool is empty: {input_path}")
    return records


def build_sft_record(record: dict[str, Any], candidate_index: int) -> dict[str, Any]:
    """Attach the frozen-pool index while preserving the ms-swift messages."""
    return {
        "candidate_index": candidate_index,
        "sample_id": record["sample_id"],
        "source": record["source"],
        "messages": record["messages"],
    }


def build_indices(
    source_size: int,
    num_groups: int,
    sample_size: int,
    base_seed: int,
    seed_step: int,
    disjoint: bool,
) -> tuple[list[list[int]], list[int]]:
    """Build one source-index list per group and return effective seeds."""
    if num_groups <= 0:
        raise ValueError("num_groups must be positive.")
    if sample_size <= 0:
        raise ValueError("sample_size must be positive.")
    if sample_size > source_size:
        raise ValueError(
            f"sample_size ({sample_size}) cannot exceed source size ({source_size})."
        )
    if disjoint and num_groups * sample_size > source_size:
        raise ValueError(
            "Disjoint sampling needs at least "
            f"{num_groups * sample_size} source records, but only {source_size} exist."
        )

    if disjoint:
        shuffled = list(range(source_size))
        random.Random(base_seed).shuffle(shuffled)
        groups = [
            shuffled[group_id * sample_size : (group_id + 1) * sample_size]
            for group_id in range(num_groups)
        ]
        return groups, [base_seed] * num_groups

    seeds = [base_seed + group_id * seed_step for group_id in range(num_groups)]
    groups = [
        random.Random(seed).sample(range(source_size), sample_size)
        for seed in seeds
    ]
    return groups, seeds


def pairwise_overlap_counts(index_groups: list[list[int]]) -> list[list[int]]:
    index_sets = [set(indices) for indices in index_groups]
    return [
        [len(left.intersection(right)) for right in index_sets]
        for left in index_sets
    ]


def sha256_file(file_path: Path) -> str:
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative_to_root(path: Path) -> str:
    resolved_path = path.resolve()
    try:
        return resolved_path.relative_to(ROOT_DIR.resolve()).as_posix()
    except ValueError:
        return str(resolved_path)


def write_json(data: Any, output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def write_jsonl(records: list[Any], output_path: Path, description: str) -> None:
    """Write one JSON record per line for ms-swift consumption."""
    with output_path.open("w", encoding="utf-8") as handle:
        for record in tqdm(records, desc=description, unit="row"):
            handle.write(json.dumps(record, ensure_ascii=False))
            handle.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate random ms-swift subsets from the frozen candidate pool. "
            "Each record keeps its candidate-pool index."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Frozen candidate_messages.jsonl (default: {DEFAULT_INPUT}).",
    )
    parser.add_argument(
        "--sft_output_dir",
        type=Path,
        default=DEFAULT_SFT_OUTPUT_DIR,
        help=f"ms-swift JSONL output directory (default: {DEFAULT_SFT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--num_groups",
        type=int,
        default=12,
        help="Number of random groups to generate (default: 12).",
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=10000,
        help="Number of records in each group (default: 10000).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Base random seed (default: 123).",
    )
    parser.add_argument(
        "--seed_step",
        type=int,
        default=1,
        help="Seed increment between independent groups (default: 1).",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="random",
        help="Output filename prefix (default: random).",
    )
    parser.add_argument(
        "--disjoint",
        action="store_true",
        help="Make groups mutually disjoint using one global shuffled pool.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Manifest path (default: <sft_output_dir>/random_manifest.json).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow existing group files and manifest to be overwritten.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = resolve_path(args.input)
    sft_output_dir = resolve_path(args.sft_output_dir)
    manifest_path = (
        resolve_path(args.manifest)
        if args.manifest is not None
        else sft_output_dir / "random_manifest.json"
    )

    records = load_candidate_records(input_path)
    index_groups, effective_seeds = build_indices(
        source_size=len(records),
        num_groups=args.num_groups,
        sample_size=args.sample_size,
        base_seed=args.seed,
        seed_step=args.seed_step,
        disjoint=args.disjoint,
    )

    planned_paths = [
        sft_output_dir / f"{args.prefix}_{group_id:02d}.jsonl"
        for group_id in range(1, args.num_groups + 1)
    ]
    planned_paths.append(manifest_path)
    existing_paths = [path for path in planned_paths if path.exists()]
    if existing_paths and not args.overwrite:
        preview = ", ".join(str(path) for path in existing_paths[:3])
        if len(existing_paths) > 3:
            preview += f", ... ({len(existing_paths)} existing files)"
        raise FileExistsError(
            f"Output already exists: {preview}. Use --overwrite or choose another output location."
        )

    sft_output_dir.mkdir(parents=True, exist_ok=True)
    group_metadata = []
    for group_id, (indices, group_seed) in enumerate(
        zip(index_groups, effective_seeds), start=1
    ):
        sft_filename = f"{args.prefix}_{group_id:02d}.jsonl"
        sft_path = sft_output_dir / sft_filename
        selected_records = [
            build_sft_record(records[index], index) for index in indices
        ]
        write_jsonl(
            selected_records,
            sft_path,
            description=f"Write {sft_filename}",
        )

        group_metadata.append(
            {
                "group_id": group_id,
                "name": f"{args.prefix}_{group_id:02d}",
                "seed": group_seed,
                "count": len(indices),
                "candidate_indices": indices,
                "sft_output": relative_to_root(sft_path),
                "sft_sha256": sha256_file(sft_path),
            }
        )

    manifest = {
        "input": relative_to_root(input_path),
        "source_format": "candidate_messages.jsonl",
        "source_count": len(records),
        "index_field": "candidate_index",
        "index_base": 0,
        "num_groups": args.num_groups,
        "sample_size": args.sample_size,
        "base_seed": args.seed,
        "seed_step": args.seed_step,
        "sampling": (
            "disjoint_without_replacement"
            if args.disjoint
            else "independent_without_replacement"
        ),
        "output_format": (
            "JSONL; each line is {candidate_index, sample_id, source, messages}"
        ),
        "groups": group_metadata,
        "pairwise_overlap_counts": pairwise_overlap_counts(index_groups),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(manifest, manifest_path)

    print(
        json.dumps(
            {
                "source_count": len(records),
                "groups_created": len(group_metadata),
                "sample_size": args.sample_size,
                "sampling": manifest["sampling"],
                "sft_output_dir": relative_to_root(sft_output_dir),
                "manifest": relative_to_root(manifest_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
