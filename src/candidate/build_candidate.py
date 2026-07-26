"""Build the frozen SCS candidate pool.

Run from the repository root with:

    python -m src.candidate.build_candidate \
        --tulu-path data/raw/tulu-3-sft-mixture \
        --coig-path data/raw/COIG-CQIA \
        --oasst-path data/raw/oasst2 \
        --minhash-tokenizer-model /path/to/tokenizer.model

The final role-aware ``doc`` is created only after source quota sampling. The
pre-candidate stages use an internal text view for decontamination and
deduplication and never export it as a document dataset.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm

if __package__ in {None, ""}:  # pragma: no cover - convenience script mode
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.candidate.export_candidate import (
    export_candidate_pool,
    export_oasst_validation,
    sha256_file,
    write_json,
    write_jsonl,
)
from src.candidate.quota import calculate_source_quotas
from src.data.coig_cqia_adapter import adapt_coig
from src.data.dedup import (
    PromptDecontaminationIndex,
    exact_deduplicate,
    minhash_deduplicate,
    sentencepiece_ngrams,
    character_ngrams,
    sha256_text,
)
from src.data.doc_builder import build_doc
from src.data.loaders import read_records
from src.data.normalize import (
    first_user_prompt,
    match_normalize,
    validate_messages,
)
from src.data.oasst2_builder import build_oasst2_validation
from src.data.tulu3_adapter import adapt_tulu


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_CANDIDATE_DIR = ROOT_DIR / "data" / "candidate" / "v1"
DEFAULT_DEV_DIR = ROOT_DIR / "data" / "dev" / "oasst2"

TULU_RELATIVE_FILES = tuple(
    Path("data") / f"train-{index:05d}-of-00006.parquet"
    for index in range(6)
)
COIG_RELATIVE_FILES = (Path("COIG-CQIA-full.jsonl"),)
OASST2_VALIDATION_RELATIVE_FILES = (
    Path("data") / "validation-00000-of-00001-1deeef95c3248fe0.parquet",
)


def increment(counter: Counter, key: str, amount: int = 1) -> None:
    counter[key] += amount


def priority(record: dict[str, Any]) -> tuple[int, str, str]:
    metadata = record.get("metadata", {})
    if record.get("source") == "coig_cqia" and metadata.get("human_verified") is True:
        level = 0
    elif record.get("source") == "coig_cqia":
        level = 1
    else:
        level = 2
    return level, str(record.get("source", "")), str(record.get("sample_id", ""))


def prepare_record(record: dict[str, Any]) -> dict[str, Any]:
    """Add transient matching fields without exporting the final doc."""

    messages = record["messages"]
    dedup_text = build_doc(messages)
    record = dict(record)
    record["_dedup_text"] = match_normalize(dedup_text)
    record["_first_user_prompt"] = match_normalize(first_user_prompt(messages))
    return record


def remove_transient(record: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in record.items()
        if not key.startswith("_")
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build SCS candidate pool v1.")
    parser.add_argument(
        "--tulu-path",
        type=Path,
        default=ROOT_DIR / "data" / "raw" / "tulu-3-sft-mixture",
    )
    parser.add_argument(
        "--coig-path",
        type=Path,
        default=ROOT_DIR / "data" / "raw" / "COIG-CQIA",
    )
    parser.add_argument(
        "--oasst-path",
        type=Path,
        default=ROOT_DIR / "data" / "raw" / "oasst2",
    )
    parser.add_argument("--candidate-dir", type=Path, default=DEFAULT_CANDIDATE_DIR)
    parser.add_argument("--dev-dir", type=Path, default=DEFAULT_DEV_DIR)
    parser.add_argument(
        "--normalized-dir",
        type=Path,
        default=ROOT_DIR / "data" / "normalized",
    )
    parser.add_argument("--target-count", type=int, default=300_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--candidate-version", default="candidate_v1")
    parser.add_argument(
        "--minhash-tokenizer-model",
        type=Path,
        default=None,
        help=(
            "SentencePiece .model, Hugging Face tokenizer directory, or model "
            "ID for the fixed document MinHash stage."
        ),
    )
    parser.add_argument(
        "--doc-minhash-tokenization",
        choices=("sentencepiece", "character"),
        default="sentencepiece",
        help="Production value is sentencepiece; character is for fixture tests only.",
    )
    parser.add_argument("--minhash-window-size", type=int, default=5)
    parser.add_argument("--minhash-permutations", type=int, default=256)
    parser.add_argument("--minhash-threshold", type=float, default=0.80)
    parser.add_argument("--prompt-threshold", type=float, default=0.80)
    parser.add_argument(
        "--skip-minhash-dedup",
        action="store_true",
        help="Skip document MinHash near-duplicate deduplication.",
    )
    parser.add_argument(
        "--skip-oasst2-decontamination",
        action="store_true",
        help="Skip decontamination against the OASST2 validation set.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_and_adapt(
    args: argparse.Namespace,
) -> tuple[list[dict], Counter, dict[str, int], int]:
    removed = Counter()
    tulu_rows = read_records(
        args.tulu_path,
        relative_paths=TULU_RELATIVE_FILES,
        split_name="train",
        description="Load Tulu train shards",
    )
    coig_rows = read_records(
        args.coig_path,
        relative_paths=COIG_RELATIVE_FILES,
        description="Load COIG-CQIA full JSONL",
    )
    tulu, tulu_removed = adapt_tulu(tulu_rows)
    coig, coig_removed = adapt_coig(coig_rows)
    for key, value in tulu_removed.items():
        increment(removed, f"tulu:{key}", value)
    for key, value in coig_removed.items():
        increment(removed, f"coig:{key}", value)

    for record in tqdm(
        tulu + coig,
        desc="Validate and prepare training records",
        unit="row",
    ):
        try:
            valid, reason = validate_messages(record["messages"])
            if not valid:
                increment(removed, f"structure:{reason}")
                record["_invalid"] = True
            else:
                record.update(prepare_record(record))
        except (TypeError, ValueError) as exc:
            increment(removed, f"structure:{exc}")
            record["_invalid"] = True
    training = [record for record in tulu + coig if not record.get("_invalid")]
    return training, removed, {
        "tulu_raw": len(tulu_rows),
        "coig_raw": len(coig_rows),
    }, len(training)


def load_dev(args: argparse.Namespace) -> tuple[list[dict], Counter]:
    rows = read_records(
        args.oasst_path,
        relative_paths=OASST2_VALIDATION_RELATIVE_FILES,
        split_name="validation",
        description="Load OASST2 validation",
    )
    records, removed = build_oasst2_validation(rows)
    for record in tqdm(records, desc="Prepare OASST2 validation", unit="row"):
        record.update(prepare_record(record))
    return records, Counter(removed)


def apply_decontamination(
    training: list[dict], dev: list[dict], removed: Counter, threshold: float
) -> tuple[list[dict], list[dict]]:
    index = PromptDecontaminationIndex(dev, threshold=threshold)
    kept = []
    removals = []
    for record in tqdm(training, desc="OASST2 decontamination", unit="row"):
        match = index.match(record)
        if match is None:
            kept.append(record)
            continue
        removals.append({"sample_id": record["sample_id"], **match})
        increment(removed, f"oasst2:{match['match_type']}")
    return kept, removals


def export_removal_records(candidate_dir: Path, name: str, records: list[dict]) -> str:
    path = candidate_dir / "removals" / f"{name}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")
    return str(path)


def apply_dedup(training: list[dict], args: argparse.Namespace, removed: Counter):
    training, exact_removed = exact_deduplicate(
        training, lambda record: record["_dedup_text"], priority
    )
    increment(removed, "train:exact_doc", len(exact_removed))

    if args.skip_minhash_dedup:
        return training, exact_removed

    if args.doc_minhash_tokenization == "sentencepiece":
        def shingle_getter(text: str):
            return sentencepiece_ngrams(
                text,
                str(args.minhash_tokenizer_model) if args.minhash_tokenizer_model else None,
                args.minhash_window_size,
            )
    else:
        def shingle_getter(text: str):
            return character_ngrams(text, args.minhash_window_size)

    training, minhash_removed = minhash_deduplicate(
        training,
        lambda record: record["_dedup_text"],
        priority,
        shingle_getter,
        num_permutations=args.minhash_permutations,
        threshold=args.minhash_threshold,
        seed=args.seed,
    )
    increment(removed, "train:minhash_doc", len(minhash_removed))
    return training, exact_removed + minhash_removed


def sample_by_source(
    records: list[dict], quotas: dict[str, int], seed: int
) -> tuple[list[dict], str]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        grouped[record["source"]].append(record)

    selected = []
    backend = "python_random"
    dj_selector = None
    dj_dataset = None
    if seed == 42:
        try:
            from datasets import Dataset
            from data_juicer.ops.selector.random_selector import RandomSelector

            dj_dataset = Dataset
            dj_selector = RandomSelector
            backend = "data_juicer.random_selector"
        except ImportError:
            pass
    for source in sorted(grouped):
        rows = sorted(grouped[source], key=lambda record: record["sample_id"])
        if dj_selector is not None and dj_dataset is not None:
            dataset = dj_dataset.from_list(rows)
            sampled = dj_selector(select_num=quotas[source]).process(dataset)
            selected.extend(sampled.to_list())
        else:
            rng = random.Random(seed)
            rng.shuffle(rows)
            selected.extend(rows[: quotas[source]])
    return selected, backend


def stage_outputs_complete(
    paths: list[Path], *, label: str, overwrite: bool
) -> bool:
    """Return whether a complete stage can be reused safely."""

    existing = [path for path in paths if path.is_file()]
    if not existing or overwrite:
        return False
    if len(existing) != len(paths):
        missing = [str(path) for path in paths if not path.is_file()]
        raise FileExistsError(
            f"{label} has partial outputs; missing:\n" + "\n".join(missing)
        )
    print(f"Skip {label}: outputs already exist")
    return True


def build(args: argparse.Namespace) -> dict[str, Any]:
    candidate_dir = args.candidate_dir.resolve()
    manifest_path = candidate_dir / "candidate_manifest.json"
    candidate_output_paths = [
        manifest_path,
        candidate_dir / "candidate.canonical.jsonl",
        candidate_dir / "candidate_messages.jsonl",
        candidate_dir / "candidate_sft.jsonl",
        candidate_dir / "candidate_doc.jsonl",
        candidate_dir / "candidate_metadata.jsonl",
        candidate_dir / "removals" / "oasst2_decontamination.jsonl",
        candidate_dir / "removals" / "training_deduplication.jsonl",
    ]
    if candidate_dir.exists() and any(candidate_dir.iterdir()) and not args.overwrite:
        if stage_outputs_complete(
            candidate_output_paths,
            label="candidate pool",
            overwrite=args.overwrite,
        ):
            return json.loads(manifest_path.read_text(encoding="utf-8"))
        raise FileExistsError(
            f"Candidate directory has partial outputs: {candidate_dir}. Use "
            "--overwrite only when intentionally rebuilding a version."
        )
    dev_dir = args.dev_dir.resolve()
    normalized_dir = args.normalized_dir.resolve()
    training, removed, raw_counts, after_adapter_count = load_and_adapt(args)
    normalized_records = []
    for record in tqdm(training, desc="Write normalized records", unit="row"):
        normalized_records.append(
            {
                "sample_id": record["sample_id"],
                "source": record["source"],
                "messages": record["messages"],
                "dedup_text": record["_dedup_text"],
                "first_user_prompt": record["_first_user_prompt"],
                "metadata": record["metadata"],
            }
        )
    normalized_by_source = defaultdict(list)
    for record in normalized_records:
        normalized_by_source[record["metadata"]["source_dataset"]].append(record)
    normalized_paths = {
        "tulu3": normalized_dir / "tulu3.jsonl",
        "coig_cqia": normalized_dir / "coig_cqia.jsonl",
        "train_merged": normalized_dir / "train_merged.jsonl",
    }
    if not stage_outputs_complete(
        list(normalized_paths.values()),
        label="normalized records",
        overwrite=args.overwrite,
    ):
        normalized_dir.mkdir(parents=True, exist_ok=True)
        write_jsonl(normalized_paths["tulu3"], normalized_by_source.get("tulu3", []))
        write_jsonl(normalized_paths["coig_cqia"], normalized_by_source.get("coig_cqia", []))
        write_jsonl(normalized_paths["train_merged"], normalized_records)
    dev, dev_removed = load_dev(args)
    removed.update(dev_removed)
    dev_hashes = [
        {
            "sample_id": record["sample_id"],
            "doc_sha256": sha256_text(record["_dedup_text"]),
            "prompt_sha256": sha256_text(record["_first_user_prompt"]),
        }
        for record in dev
    ]

    if args.skip_oasst2_decontamination:
        decontamination_removed = []
    else:
        training, decontamination_removed = apply_decontamination(
            training, dev, removed, args.prompt_threshold
        )
    after_decontamination_count = len(training)
    training, dedup_removed = apply_dedup(training, args, removed)
    after_dedup_count = len(training)
    source_counts = Counter(record["source"] for record in training)
    quotas = calculate_source_quotas(source_counts, args.target_count)
    selected, selection_backend = sample_by_source(training, quotas, args.seed)
    if len(selected) != args.target_count:
        raise AssertionError("selected candidate count does not equal target")

    # The role-aware doc is materialized only for the frozen candidate rows.
    for record in tqdm(selected, desc="Build candidate docs", unit="row"):
        record["doc"] = build_doc(record["messages"])
    selected = [remove_transient(record) for record in selected]

    dev_hash_manifest = dev_dir / "oasst2_hashes.jsonl"
    dev_output_paths = [
        dev_dir / "oasst2_validation.canonical.jsonl",
        dev_dir / "oasst2_validation.jsonl",
        dev_hash_manifest,
    ]
    if stage_outputs_complete(
        dev_output_paths,
        label="OASST2 validation outputs",
        overwrite=args.overwrite,
    ):
        dev_hash_files = {
            "oasst2_validation_canonical": sha256_file(dev_output_paths[0]),
            "oasst2_validation": sha256_file(dev_output_paths[1]),
        }
    else:
        dev_dir.mkdir(parents=True, exist_ok=True)
        with dev_hash_manifest.open("w", encoding="utf-8", newline="\n") as handle:
            for row in dev_hashes:
                handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
                handle.write("\n")
        dev_hash_files = export_oasst_validation(dev_dir, dev)
    candidate_hashes = export_candidate_pool(candidate_dir, selected)
    decontamination_log = export_removal_records(
        candidate_dir, "oasst2_decontamination", decontamination_removed
    )
    dedup_log = export_removal_records(
        candidate_dir, "training_deduplication", dedup_removed
    )

    language_counts = Counter(
        str(record.get("metadata", {}).get("language", "unknown")) for record in selected
    )
    manifest = {
        "candidate_version": args.candidate_version,
        "sample_count": len(selected),
        "random_seed": args.seed,
        "doc_format_version": "role_doc_v1",
        "dedup": {
            "text_key": "transient_role_doc",
            "exact": True,
            "minhash_enabled": not args.skip_minhash_dedup,
            "tokenization": args.doc_minhash_tokenization,
            "window_size": args.minhash_window_size,
            "num_permutations": args.minhash_permutations,
            "jaccard_threshold": args.minhash_threshold,
        },
        "oasst2_decontamination": {
            "enabled": not args.skip_oasst2_decontamination,
            "exact_doc": True,
            "exact_prompt": True,
            "prompt_character_5gram_jaccard_threshold": args.prompt_threshold,
        },
        "source_distribution": dict(sorted(Counter(record["source"] for record in selected).items())),
        "language_distribution": dict(sorted(language_counts.items())),
        "removed_statistics": dict(sorted(removed.items())),
        "stage_counts": {
            **raw_counts,
            "oasst2_validation_records": len(dev),
            "training_after_adapters": after_adapter_count,
            "training_after_decontamination": after_decontamination_count,
            "training_after_dedup": after_dedup_count,
            "candidate": len(selected),
        },
        "quota": quotas,
        "selection_backend": selection_backend,
        "file_sha256": {
            **dev_hash_files,
            **candidate_hashes,
            "oasst2_hashes": sha256_file(dev_hash_manifest),
            **{
                f"normalized_{name}": sha256_file(path)
                for name, path in normalized_paths.items()
            },
        },
        "removal_records": {
            "oasst2_decontamination": decontamination_log,
            "training_deduplication": dedup_log,
        },
    }
    write_json(manifest_path, manifest)
    return manifest


def main() -> None:
    args = parse_args()
    manifest = build(args)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
