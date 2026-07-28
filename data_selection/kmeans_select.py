"""Generate K-Means subsets from the frozen candidate pool."""

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from tqdm.auto import tqdm


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT_DIR / "data" / "candidate" / "v1" / "candidate_messages.jsonl"
DEFAULT_EMBEDDING_NPZ = (
    ROOT_DIR / "output" / "feature" / "candidate_embeddings_llama.npz"
)
DEFAULT_SFT_OUTPUT_DIR = ROOT_DIR / "data" / "sft" / "kmeans"


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT_DIR / path


def load_candidate_records(input_path: Path) -> list[dict[str, Any]]:
    """Load and validate the frozen candidate messages in source order."""
    if not input_path.exists():
        raise FileNotFoundError(f"Candidate pool not found: {input_path}")
    if input_path.suffix.lower() != ".jsonl":
        raise ValueError("K-Means selection expects candidate_messages.jsonl.")

    records = []
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
                    f"Invalid JSON at line {line_number}: {exc}"
                ) from exc
            if not isinstance(record, dict):
                raise ValueError(f"Candidate record {line_number} must be an object.")

            for field in ("sample_id", "source"):
                if not isinstance(record.get(field), str) or not record[field].strip():
                    raise ValueError(
                        f"Candidate record {line_number} needs a non-empty '{field}'."
                    )
            messages = record.get("messages")
            if not isinstance(messages, list) or not messages:
                raise ValueError(
                    f"Candidate record {line_number} needs a non-empty messages list."
                )
            for message in messages:
                if not isinstance(message, dict):
                    raise ValueError(
                        f"Candidate record {line_number} has an invalid message."
                    )
                if not isinstance(message.get("role"), str) or not isinstance(
                    message.get("content"), str
                ):
                    raise ValueError(
                        f"Candidate record {line_number} has an invalid message role/content."
                    )

            sample_id = record["sample_id"]
            if sample_id in sample_ids:
                raise ValueError(f"Duplicate candidate sample_id: {sample_id}")
            sample_ids.add(sample_id)
            records.append(record)

    if not records:
        raise ValueError(f"Candidate pool is empty: {input_path}")
    return records


def load_embeddings(
    embedding_path: Path,
    records: list[dict[str, Any]],
) -> np.ndarray:
    """Load embeddings and verify their row alignment with the candidate pool."""
    if not embedding_path.exists():
        raise FileNotFoundError(f"Embedding NPZ not found: {embedding_path}")

    with np.load(embedding_path, allow_pickle=False) as data:
        if "embeddings" not in data or "sample_ids" not in data:
            raise ValueError(
                "Embedding NPZ must contain 'embeddings' and 'sample_ids'."
            )
        embeddings = np.asarray(data["embeddings"], dtype=np.float32)
        sample_ids = np.asarray(data["sample_ids"]).astype(str).tolist()

    expected_ids = [record["sample_id"] for record in records]
    if sample_ids != expected_ids:
        raise ValueError(
            "Embedding/candidate alignment mismatch: sample_ids must match "
            "candidate_messages.jsonl row order."
        )
    if embeddings.ndim != 2 or embeddings.shape[0] != len(records):
        raise ValueError(
            "Embedding shape mismatch: expected a 2-D matrix with one row per "
            f"candidate ({len(records)}), got {embeddings.shape}."
        )
    if embeddings.shape[1] == 0:
        raise ValueError("Embedding vectors must not be empty.")
    if not np.isfinite(embeddings).all():
        raise ValueError("Embedding NPZ contains NaN or infinity values.")
    return embeddings


def build_indices(
    labels: np.ndarray,
    num_groups: int,
    sample_size: int,
    base_seed: int,
    seed_step: int,
) -> tuple[list[list[int]], list[int]]:
    """Sample the same number of records from every K-Means cluster."""
    source_size = len(labels)
    if num_groups <= 0:
        raise ValueError("num_groups must be positive.")
    if sample_size <= 0 or sample_size > source_size:
        raise ValueError(
            f"sample_size must be between 1 and {source_size}, got {sample_size}."
        )

    cluster_map: dict[int, list[int]] = {}
    for index, label in enumerate(labels.tolist()):
        cluster_map.setdefault(int(label), []).append(index)

    quota = sample_size // len(cluster_map)
    seeds = [base_seed + group_id * seed_step for group_id in range(num_groups)]
    index_groups = []
    for seed in seeds:
        rng = random.Random(seed)
        selected: list[int] = []
        for label in sorted(cluster_map):
            members = list(cluster_map[label])
            rng.shuffle(members)
            selected.extend(members[:quota])

        selected_set = set(selected)
        remaining = [index for index in range(source_size) if index not in selected_set]
        rng.shuffle(remaining)
        selected.extend(remaining[: sample_size - len(selected)])
        index_groups.append(selected)

    return index_groups, seeds


def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """Normalize rows in place to avoid copying the full embedding matrix."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    np.divide(embeddings, norms, out=embeddings, where=norms != 0)
    return embeddings


def choose_k_values(
    k_min: int,
    k_max: int,
    groups: int,
    seed: int,
) -> list[int]:
    """Choose one deterministic random K from each evenly spaced interval."""
    if k_min < 20 or k_max > 100 or k_min > k_max:
        raise ValueError("k_range must be within 20..100 and min <= max.")
    if groups <= 0:
        raise ValueError("groups must be positive.")
    interval_size = k_max - k_min + 1
    if groups > interval_size:
        raise ValueError(
            "groups cannot exceed the number of integer K values in k_range."
        )

    rng = random.Random(seed)
    values = []
    for group_id in range(groups):
        start = k_min + interval_size * group_id // groups
        end = k_min + interval_size * (group_id + 1) // groups - 1
        values.append(rng.randint(start, end))
    rng.shuffle(values)
    return values


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
    try:
        return path.resolve().relative_to(ROOT_DIR.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def write_json(data: Any, output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def write_jsonl(records: list[dict[str, Any]], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as handle:
        for record in tqdm(records, desc=f"Write {output_path.name}", unit="row"):
            handle.write(json.dumps(record, ensure_ascii=False))
            handle.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate K-Means ms-swift subsets from candidate embeddings."
    )
    parser.add_argument(
        "--input",
        "--candidate-path",
        "--candidate_path",
        dest="input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Frozen candidate_messages.jsonl (default: {DEFAULT_INPUT}).",
    )
    parser.add_argument(
        "--embedding-npz",
        "--embedding_npz",
        "--npz",
        dest="embedding_npz",
        type=Path,
        default=DEFAULT_EMBEDDING_NPZ,
        help=f"Candidate embedding NPZ (default: {DEFAULT_EMBEDDING_NPZ}).",
    )
    parser.add_argument(
        "--sft-output-dir",
        "--sft_output_dir",
        dest="sft_output_dir",
        type=Path,
        default=DEFAULT_SFT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_SFT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--sample-size",
        "--sample_size",
        dest="sample_size",
        type=int,
        default=10000,
        help="Number of records in each group (default: 10000).",
    )
    parser.add_argument(
        "--k-range",
        "--k_range",
        dest="k_range",
        type=int,
        nargs=2,
        default=(50, 100),
        metavar=("MIN", "MAX"),
        help="Inclusive K range (default: 50 100; allowed range: 20 100).",
    )
    parser.add_argument(
        "--groups",
        "--num-groups",
        "--num_groups",
        dest="groups",
        type=int,
        default=12,
        help="Number of K-Means groups to generate (default: 12).",
    )
    parser.add_argument(
        "--batch-size",
        "--batch_size",
        dest="batch_size",
        type=int,
        default=4096,
        help="MiniBatchKMeans batch size (default: 4096).",
    )
    parser.add_argument(
        "--max-iter",
        "--max_iter",
        dest="max_iter",
        type=int,
        default=100,
        help="MiniBatchKMeans iterations (default: 100).",
    )
    parser.add_argument(
        "--n-init",
        "--n_init",
        dest="n_init",
        type=int,
        default=3,
        help="Number of MiniBatchKMeans initializations (default: 3).",
    )
    parser.add_argument("--seed", type=int, default=46)
    parser.add_argument(
        "--prefix",
        default=None,
        help="Output prefix (default: kmeans).",
    )
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing group files and manifest.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.max_iter <= 0 or args.n_init <= 0:
        raise ValueError("batch_size, max_iter, and n_init must be positive.")
    k_min, k_max = args.k_range
    k_values = choose_k_values(k_min, k_max, args.groups, args.seed)

    input_path = resolve_path(args.input)
    embedding_path = resolve_path(args.embedding_npz)
    output_dir = resolve_path(args.sft_output_dir)
    prefix = args.prefix or "kmeans"
    manifest_path = (
        resolve_path(args.manifest)
        if args.manifest is not None
        else output_dir / f"{prefix}_manifest.json"
    )

    records = load_candidate_records(input_path)
    embeddings = load_embeddings(embedding_path, records)
    if k_max > len(records):
        raise ValueError(f"k_max ({k_max}) cannot exceed source size ({len(records)}).")
    if args.batch_size < k_max:
        raise ValueError("batch_size must be at least k_max.")

    output_paths = [
        output_dir / f"{prefix}_{group_id:02d}.jsonl"
        for group_id in range(1, args.groups + 1)
    ] + [manifest_path]
    existing_paths = [path for path in output_paths if path.exists()]
    if existing_paths and not args.overwrite:
        raise FileExistsError(
            f"Output already exists: {existing_paths[0]}. Use --overwrite to replace it."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    normalize_embeddings(embeddings)
    index_groups = []
    group_metadata = []
    for group_id, k in enumerate(k_values, start=1):
        group_seed = args.seed + group_id - 1
        print(
            f"Fit MiniBatchKMeans: group={group_id}, k={k}, "
            f"rows={len(records)}, batch_size={args.batch_size}"
        )
        labels = MiniBatchKMeans(
            n_clusters=k,
            batch_size=args.batch_size,
            max_iter=args.max_iter,
            n_init=args.n_init,
            random_state=group_seed,
        ).fit_predict(embeddings)
        group_indices, _ = build_indices(
            labels, 1, args.sample_size, group_seed, 1
        )
        indices = group_indices[0]
        index_groups.append(indices)

        output_path = output_dir / f"{prefix}_{group_id:02d}.jsonl"
        selected = [
            {
                "candidate_index": index,
                "sample_id": records[index]["sample_id"],
                "source": records[index]["source"],
                "messages": records[index]["messages"],
            }
            for index in indices
        ]
        write_jsonl(selected, output_path)
        group_metadata.append(
            {
                "group_id": group_id,
                "name": f"{prefix}_{group_id:02d}",
                "k": k,
                "seed": group_seed,
                "count": len(indices),
                "candidate_indices": indices,
                "sft_output": relative_to_root(output_path),
                "sft_sha256": sha256_file(output_path),
            }
        )

    manifest = {
        "input": relative_to_root(input_path),
        "embedding_npz": relative_to_root(embedding_path),
        "source_count": len(records),
        "index_field": "candidate_index",
        "index_base": 0,
        "k_range": [k_min, k_max],
        "selected_k_values": k_values,
        "group_count": args.groups,
        "batch_size": args.batch_size,
        "max_iter": args.max_iter,
        "n_init": args.n_init,
        "num_groups": args.groups,
        "sample_size": args.sample_size,
        "base_seed": args.seed,
        "sampling": "cluster_balanced_with_random_fallback",
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
                "k_range": [k_min, k_max],
                "selected_k_values": k_values,
                "batch_size": args.batch_size,
                "sft_output_dir": relative_to_root(output_dir),
                "manifest": relative_to_root(manifest_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
