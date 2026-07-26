"""Materialize aligned candidate-pool outputs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in tqdm(rows, desc=f"Write {path.name}", unit="row"):
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export_oasst_validation(
    output_dir: Path, records: list[dict[str, Any]]
) -> dict[str, str]:
    canonical_path = output_dir / "oasst2_validation.canonical.jsonl"
    messages_path = output_dir / "oasst2_validation.jsonl"
    write_jsonl(
        canonical_path,
        [
            {
                "sample_id": record["sample_id"],
                "source": record["source"],
                "messages": record["messages"],
                "metadata": record["metadata"],
            }
            for record in records
        ],
    )
    write_jsonl(messages_path, [{"messages": record["messages"]} for record in records])
    return {
        "oasst2_validation_canonical": sha256_file(canonical_path),
        "oasst2_validation": sha256_file(messages_path),
    }


def export_candidate_pool(
    output_dir: Path,
    records: list[dict[str, Any]],
) -> dict[str, str]:
    canonical_path = output_dir / "candidate.canonical.jsonl"
    messages_path = output_dir / "candidate_messages.jsonl"
    sft_path = output_dir / "candidate_sft.jsonl"
    doc_path = output_dir / "candidate_doc.jsonl"
    metadata_path = output_dir / "candidate_metadata.jsonl"

    canonical = []
    messages = []
    sft = []
    docs = []
    metadata = []
    for row_index, record in enumerate(
        tqdm(records, desc="Materialize candidate outputs", unit="row")
    ):
        common = {
            "sample_id": record["sample_id"],
            "source": record["source"],
            "messages": record["messages"],
        }
        record_metadata = dict(record["metadata"])
        record_metadata["row_index"] = row_index
        canonical.append(
            {
                **common,
                "doc": record["doc"],
                "metadata": record_metadata,
            }
        )
        messages.append(common)
        sft.append({"messages": record["messages"]})
        docs.append(
            {
                "sample_id": record["sample_id"],
                "source": record["source"],
                "doc": record["doc"],
            }
        )
        metadata.append(
            {
                "sample_id": record["sample_id"],
                "source": record["source"],
                "row_index": row_index,
                **record_metadata,
            }
        )

    write_jsonl(canonical_path, canonical)
    write_jsonl(messages_path, messages)
    write_jsonl(sft_path, sft)
    write_jsonl(doc_path, docs)
    write_jsonl(metadata_path, metadata)
    return {
        "candidate_canonical": sha256_file(canonical_path),
        "candidate_messages": sha256_file(messages_path),
        "candidate_sft": sha256_file(sft_path),
        "candidate_doc": sha256_file(doc_path),
        "candidate_metadata": sha256_file(metadata_path),
    }
