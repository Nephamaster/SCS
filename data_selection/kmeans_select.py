"""Generate reproducible K-Means subsets in SCS-doc and ms-swift formats.

The feature database is produced by ``src/utils.py``.  Its expected schema is
one table named after the database file (with ``-`` replaced by ``_``), with
the columns ``doc_id``, ``embedding`` and ``ln_probability``.  The database is
validated against the source dataset before K-Means is run so that a feature
file from another dataset cannot silently select the wrong records.

Example:
    python data_selection/kmeans_select.py \
        --input data/SFT.json \
        --feature_db output/feature/qSFT.db \
        --num_groups 12 \
        --sample_size 10000 \
        --n_clusters 50 \
        --seed 123
"""

import argparse
import hashlib
import json
import pickle
import random
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT_DIR / "data" / "SFT.json"
DEFAULT_FEATURE_DB = ROOT_DIR / "output" / "feature" / "SFT.db"
DEFAULT_DOC_OUTPUT_DIR = ROOT_DIR / "data" / "doc" / "kmeans"
DEFAULT_SFT_OUTPUT_DIR = ROOT_DIR / "data" / "sft" / "kmeans"
DEFAULT_MANIFEST = ROOT_DIR / "data" / "kmeans_manifest.json"

ROLE_MAP = {
    "human": "user",
    "user": "user",
    "gpt": "assistant",
    "assistant": "assistant",
    "bot": "assistant",
    "model": "assistant",
    "system": "system",
    "tool": "tool",
    "function": "function",
}
SUPPORTED_ROLES = set(ROLE_MAP.values())
REQUIRED_FEATURE_COLUMNS = {"doc_id", "embedding", "ln_probability"}


def resolve_path(path: Path) -> Path:
    """Resolve relative CLI paths against the repository root."""
    return path if path.is_absolute() else ROOT_DIR / path


def load_records(input_path: Path) -> list[Any]:
    """Load a JSON array or JSONL dataset."""
    if not input_path.exists():
        raise FileNotFoundError(f"Input dataset not found: {input_path}")

    with input_path.open("r", encoding="utf-8") as handle:
        if input_path.suffix.lower() == ".jsonl":
            records = [json.loads(line) for line in handle if line.strip()]
        else:
            records = json.load(handle)

    if not isinstance(records, list):
        raise ValueError(
            f"Input dataset must contain a JSON array or JSONL records: {input_path}"
        )
    return records


def normalize_role(raw_role: Any, record_index: int) -> str:
    if not isinstance(raw_role, str) or not raw_role.strip():
        raise ValueError(f"Record {record_index} has an invalid conversation role.")

    role = ROLE_MAP.get(raw_role.strip().lower(), raw_role.strip().lower())
    if role not in SUPPORTED_ROLES:
        raise ValueError(
            f"Record {record_index} uses unsupported role '{raw_role}'. "
            f"Supported roles: {sorted(SUPPORTED_ROLES)}"
        )
    return role


def normalize_messages(record: dict[str, Any], record_index: int) -> list[dict[str, str]]:
    """Convert ShareGPT conversations or existing messages to ms-swift format."""
    if isinstance(record.get("messages"), list):
        turns = record["messages"]
        source_role_key = "role"
        source_content_key = "content"
    elif isinstance(record.get("conversations"), list):
        turns = record["conversations"]
        source_role_key = "from"
        source_content_key = "value"
    else:
        raise ValueError(
            f"Record {record_index} must contain 'conversations' or 'messages'."
        )

    messages = []
    for turn_index, turn in enumerate(turns):
        if not isinstance(turn, dict):
            raise ValueError(
                f"Record {record_index}, turn {turn_index} is not an object."
            )

        content = turn.get(source_content_key, "")
        if not isinstance(content, str):
            raise ValueError(
                f"Record {record_index}, turn {turn_index} content must be a string."
            )

        messages.append(
            {
                "role": normalize_role(turn.get(source_role_key), record_index),
                "content": content,
            }
        )
    return messages


def build_doc_record(
    messages: list[dict[str, str]], tokenizer: Any
) -> dict[str, Any]:
    """Build the flattened SCS document structure."""
    # Match DatasetAnalyzer.flatten(): concatenate turns without separators.
    doc = "".join(message["content"] for message in messages)
    return {
        "doc": doc,
        "n_tokens": len(tokenizer.tokenize(doc)),
    }


def quote_identifier(identifier: str) -> str:
    """Quote an SQLite identifier safely."""
    return '"' + identifier.replace('"', '""') + '"'


def expected_table_name(feature_db: Path) -> str:
    return feature_db.stem.replace("-", "_")


def read_feature_embeddings(
    feature_db: Path,
    expected_count: int,
    invalid_embedding_policy: str = "error",
) -> tuple[np.ndarray, str, int]:
    """Validate and read the feature DB used by the selector.

    The feature extractor writes one row per source record, with autoincrement
    IDs starting at one.  Both the schema and this alignment contract are
    checked before any clustering or output file is created.
    """
    if invalid_embedding_policy not in {"error", "mean", "zero"}:
        raise ValueError(
            "invalid_embedding_policy must be one of: error, mean, zero."
        )
    if not feature_db.exists():
        raise FileNotFoundError(f"Feature database not found: {feature_db}")
    if expected_count <= 0:
        raise ValueError("The input dataset must contain at least one record.")

    table_name = expected_table_name(feature_db)
    table_name = 'SFT'
    conn = sqlite3.connect(str(feature_db))
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table' AND name NOT LIKE 'sqlite_%'
            ORDER BY name
            """
        )
        tables = [row[0] for row in cursor.fetchall()]
        if table_name not in tables:
            raise ValueError(
                f"Feature DB schema mismatch for {feature_db}: expected table "
                f"'{table_name}', found {tables or 'no user tables'}."
            )

        quoted_table = quote_identifier(table_name)
        cursor.execute(f"PRAGMA table_info({quoted_table})")
        columns = cursor.fetchall()
        column_names = {row[1].lower() for row in columns}
        missing = sorted(REQUIRED_FEATURE_COLUMNS - column_names)
        if missing:
            actual = [row[1] for row in columns]
            raise ValueError(
                f"Feature DB schema mismatch for table '{table_name}': missing "
                f"columns {missing}; actual columns are {actual}."
            )

        column_by_name = {row[1].lower(): row for row in columns}
        if column_by_name["doc_id"][5] != 1:
            raise ValueError(
                f"Feature DB schema mismatch for table '{table_name}': "
                "'doc_id' must be the primary key used to align features."
            )

        cursor.execute(f"SELECT COUNT(*) FROM {quoted_table}")
        row_count = int(cursor.fetchone()[0])
        if row_count != expected_count:
            raise ValueError(
                f"Feature/source size mismatch: table '{table_name}' has "
                f"{row_count} rows, but the input dataset has {expected_count} records."
            )

        cursor.execute(
            f"SELECT doc_id, embedding, ln_probability FROM {quoted_table} "
            "ORDER BY doc_id"
        )
        rows = cursor.fetchall()
    finally:
        conn.close()

    doc_ids = [row[0] for row in rows]
    expected_doc_ids = list(range(1, expected_count + 1))
    if doc_ids != expected_doc_ids:
        raise ValueError(
            f"Feature DB alignment mismatch: expected doc_id values 1..{expected_count} "
            f"in order, found {doc_ids[:5]}{'...' if len(doc_ids) > 5 else ''}."
        )

    embeddings = []
    invalid_doc_ids = []
    embedding_shape = None
    for row_index, (_doc_id, embedding_blob, ln_probability) in enumerate(rows):
        try:
            embedding = pickle.loads(embedding_blob)
        except Exception as exc:
            raise ValueError(
                f"Feature DB row {row_index + 1} has an unreadable embedding BLOB."
            ) from exc

        array = np.asarray(embedding)
        if array.ndim != 1 or array.size == 0:
            raise ValueError(
                f"Feature DB row {row_index + 1} embedding must be a non-empty "
                f"1-D array, got shape {array.shape}."
            )
        if not np.issubdtype(array.dtype, np.number):
            raise ValueError(
                f"Feature DB row {row_index + 1} embedding must be numeric, "
                f"got dtype {array.dtype}."
            )
        array = array.astype(np.float32, copy=False)
        if not np.isfinite(array).all():
            if invalid_embedding_policy == "error":
                raise ValueError(
                    f"Feature DB row {row_index + 1} (doc_id={_doc_id}) embedding "
                    "contains NaN or infinity. Use "
                    "--invalid_embedding_policy mean or zero to continue with "
                    "explicit repair."
                )
            invalid_doc_ids.append(int(_doc_id))
        if embedding_shape is None:
            embedding_shape = array.shape
        elif array.shape != embedding_shape:
            raise ValueError(
                f"Feature DB embedding shape mismatch at row {row_index + 1}: "
                f"expected {embedding_shape}, got {array.shape}."
            )

        try:
            probability = float(ln_probability)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Feature DB row {row_index + 1} has a non-numeric ln_probability."
            ) from exc
        if not np.isfinite(probability):
            raise ValueError(
                f"Feature DB row {row_index + 1} ln_probability is not finite."
            )
        embeddings.append(array)

    matrix = np.asarray(embeddings, dtype=np.float32)
    if invalid_doc_ids:
        finite_mask = np.isfinite(matrix)
        if invalid_embedding_policy == "zero":
            fill_values = np.zeros(matrix.shape[1], dtype=np.float32)
        else:
            # Replace only the invalid coordinates with the corresponding
            # column mean from valid coordinates.  This keeps the feature
            # dimensionality and row/source alignment intact while avoiding
            # the near-zero vectors produced by a blanket nan_to_num call.
            matrix64 = matrix.astype(np.float64, copy=False)
            valid_values = np.where(finite_mask, matrix64, 0.0)
            valid_counts = finite_mask.sum(axis=0)
            valid_sums = valid_values.sum(axis=0)
            fill_values = np.divide(
                valid_sums,
                valid_counts,
                out=np.zeros(matrix.shape[1], dtype=np.float64),
                where=valid_counts > 0,
            ).astype(np.float32)

        bad_rows, bad_columns = np.where(~finite_mask)
        matrix[bad_rows, bad_columns] = fill_values[bad_columns]
        print(
            "WARNING: repaired non-finite embeddings in "
            f"{len(invalid_doc_ids)} row(s) using policy "
            f"'{invalid_embedding_policy}'; doc_id values: "
            f"{invalid_doc_ids[:20]}"
            f"{'...' if len(invalid_doc_ids) > 20 else ''}."
        )
    print(
        f"Feature DB schema validation passed: table={table_name}, "
        f"rows={row_count}, embedding_shape={embedding_shape}."
    )
    return matrix, table_name, len(invalid_doc_ids)


def build_indices(
    labels: np.ndarray,
    num_groups: int,
    sample_size: int,
    base_seed: int,
    seed_step: int,
) -> tuple[list[list[int]], list[int]]:
    """Select balanced K-Means representatives for each independent group."""
    source_size = len(labels)
    if num_groups <= 0:
        raise ValueError("num_groups must be positive.")
    if sample_size <= 0:
        raise ValueError("sample_size must be positive.")
    if sample_size > source_size:
        raise ValueError(
            f"sample_size ({sample_size}) cannot exceed source size ({source_size})."
        )

    cluster_map: dict[int, list[int]] = {}
    for index, label in enumerate(labels.tolist()):
        cluster_map.setdefault(int(label), []).append(index)

    quota = sample_size // len(cluster_map)
    index_groups = []
    seeds = [base_seed + group_id * seed_step for group_id in range(num_groups)]
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
        if len(selected) != sample_size:
            raise RuntimeError(
                f"K-Means selection produced {len(selected)} records; "
                f"expected {sample_size}."
            )
        index_groups.append(selected)

    return index_groups, seeds


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


def write_jsonl(records: list[Any], output_path: Path) -> None:
    """Write one JSON record per line for ms-swift consumption."""
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False))
            handle.write("\n")


def pairwise_overlap_counts(index_groups: list[list[int]]) -> list[list[int]]:
    index_sets = [set(indices) for indices in index_groups]
    return [
        [len(left.intersection(right)) for right in index_sets]
        for left in index_sets
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate K-Means subsets in SCS doc format and ms-swift "
            "messages format."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Source ShareGPT JSON/JSONL dataset (default: {DEFAULT_INPUT}).",
    )
    parser.add_argument(
        "--feature_db",
        "--db",
        dest="feature_db",
        type=Path,
        default=DEFAULT_FEATURE_DB,
        help=f"SQLite feature database (default: {DEFAULT_FEATURE_DB}).",
    )
    parser.add_argument(
        "--doc_output_dir",
        type=Path,
        default=DEFAULT_DOC_OUTPUT_DIR,
        help=f"SCS doc output directory (default: {DEFAULT_DOC_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--sft_output_dir",
        type=Path,
        default=DEFAULT_SFT_OUTPUT_DIR,
        help=f"ms-swift SFT output directory (default: {DEFAULT_SFT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--num_groups",
        "--num_datasets",
        "--num_dataset",
        dest="num_groups",
        type=int,
        default=1,
        help="Number of K-Means subsets to generate (default: 1).",
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=10000,
        help="Number of records in each subset (default: 10000).",
    )
    parser.add_argument(
        "--n_clusters",
        type=int,
        default=50,
        help="Number of K-Means clusters (default: 50).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=46,
        help="K-Means and first subset seed (default: 46).",
    )
    parser.add_argument(
        "--seed_step",
        type=int,
        default=1,
        help="Seed increment between independently sampled subsets (default: 1).",
    )
    parser.add_argument(
        "--tokenize_model",
        type=str,
        default="FacebookAI/xlm-roberta-large",
        help="Tokenizer used for doc.n_tokens, matching datastation.py.",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="kmeans",
        help="Output filename prefix (default: kmeans).",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help=f"Manifest path (default: {DEFAULT_MANIFEST}).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow existing subset files and manifest to be overwritten.",
    )
    parser.add_argument(
        "--invalid_embedding_policy",
        choices=("mean", "zero", "error"),
        default="mean",
        help=(
            "How to handle non-finite embedding values: replace them with "
            "per-dimension valid-row means, replace with zero, or fail. "
            "Default: mean."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = resolve_path(args.input)
    feature_db = resolve_path(args.feature_db)
    doc_output_dir = resolve_path(args.doc_output_dir)
    sft_output_dir = resolve_path(args.sft_output_dir)
    manifest_path = resolve_path(args.manifest)

    records = load_records(input_path)
    if not records:
        raise ValueError(f"Input dataset is empty: {input_path}")
    # Validate the database before loading a tokenizer, fitting K-Means, or
    # creating outputs.  This is the critical source/feature alignment gate.
    embeddings, table_name, repaired_embedding_count = read_feature_embeddings(
        feature_db,
        expected_count=len(records),
        invalid_embedding_policy=args.invalid_embedding_policy,
    )

    if args.n_clusters <= 0:
        raise ValueError("n_clusters must be positive.")
    if args.n_clusters > len(records):
        raise ValueError(
            f"n_clusters ({args.n_clusters}) cannot exceed source size ({len(records)})."
        )

    normalized_embeddings = normalize(embeddings, axis=1)
    kmeans = KMeans(
        n_clusters=args.n_clusters,
        n_init=15,
        random_state=args.seed,
    )
    labels = kmeans.fit_predict(normalized_embeddings)
    index_groups, effective_seeds = build_indices(
        labels=labels,
        num_groups=args.num_groups,
        sample_size=args.sample_size,
        base_seed=args.seed,
        seed_step=args.seed_step,
    )

    planned_paths = []
    for group_id in range(1, args.num_groups + 1):
        doc_filename = f"{args.prefix}_{group_id:02d}.json"
        sft_filename = f"{args.prefix}_{group_id:02d}.jsonl"
        planned_paths.extend(
            [doc_output_dir / doc_filename, sft_output_dir / sft_filename]
        )
    planned_paths.append(manifest_path)
    existing_paths = [path for path in planned_paths if path.exists()]
    if existing_paths and not args.overwrite:
        preview = ", ".join(str(path) for path in existing_paths[:3])
        if len(existing_paths) > 3:
            preview += f", ... ({len(existing_paths)} existing files)"
        raise FileExistsError(
            f"Output already exists: {preview}. Use --overwrite or choose another output location."
        )

    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Creating SCS doc files requires transformers. Install the project "
            "requirements or provide an environment with AutoTokenizer."
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenize_model,
        clean_up_tokenization_spaces=False,
        model_max_length=4096,
    )

    needed_indices = sorted({index for group in index_groups for index in group})
    sft_cache: dict[int, list[dict[str, str]]] = {}
    doc_cache: dict[int, dict[str, Any]] = {}
    for index in needed_indices:
        record = records[index]
        if not isinstance(record, dict):
            raise ValueError(f"Record {index} must be an object.")
        messages = normalize_messages(record, index)
        sft_cache[index] = messages
        doc_cache[index] = build_doc_record(messages, tokenizer)

    doc_output_dir.mkdir(parents=True, exist_ok=True)
    sft_output_dir.mkdir(parents=True, exist_ok=True)
    group_metadata = []
    for group_id, (indices, group_seed) in enumerate(
        zip(index_groups, effective_seeds), start=1
    ):
        doc_filename = f"{args.prefix}_{group_id:02d}.json"
        sft_filename = f"{args.prefix}_{group_id:02d}.jsonl"
        doc_path = doc_output_dir / doc_filename
        sft_path = sft_output_dir / sft_filename

        write_json([doc_cache[index] for index in indices], doc_path)
        write_jsonl(
            [{"messages": sft_cache[index]} for index in indices], sft_path
        )

        group_metadata.append(
            {
                "group_id": group_id,
                "name": f"{args.prefix}_{group_id:02d}",
                "seed": group_seed,
                "count": len(indices),
                "doc_output": relative_to_root(doc_path),
                "sft_output": relative_to_root(sft_path),
                "doc_sha256": sha256_file(doc_path),
                "sft_sha256": sha256_file(sft_path),
            }
        )

    manifest = {
        "input": relative_to_root(input_path),
        "feature_database": relative_to_root(feature_db),
        "feature_table": table_name,
        "source_count": len(records),
        "num_groups": args.num_groups,
        "sample_size": args.sample_size,
        "n_clusters": args.n_clusters,
        "base_seed": args.seed,
        "seed_step": args.seed_step,
        "tokenize_model": args.tokenize_model,
        "invalid_embedding_policy": args.invalid_embedding_policy,
        "repaired_embedding_rows": repaired_embedding_count,
        "sampling": "cluster_balanced_with_random_fallback",
        "formats": {
            "doc": '{"doc": string, "n_tokens": integer}',
            "sft": 'JSONL; each line is {"messages": [{"role": string, "content": string}, ...]}',
        },
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
                "n_clusters": args.n_clusters,
                "repaired_embedding_rows": repaired_embedding_count,
                "doc_output_dir": relative_to_root(doc_output_dir),
                "sft_output_dir": relative_to_root(sft_output_dir),
                "manifest": relative_to_root(manifest_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
