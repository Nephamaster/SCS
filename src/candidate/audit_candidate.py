"""Audit frozen candidate-pool alignment and message invariants."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.doc_builder import build_doc
from src.data.normalize import validate_messages


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def audit(candidate_dir: Path, expected_count: int | None = None) -> dict[str, Any]:
    canonical = read_jsonl(candidate_dir / "candidate.canonical.jsonl")
    messages = read_jsonl(candidate_dir / "candidate_messages.jsonl")
    sft = read_jsonl(candidate_dir / "candidate_sft.jsonl")
    docs = read_jsonl(candidate_dir / "candidate_doc.jsonl")
    metadata = read_jsonl(candidate_dir / "candidate_metadata.jsonl")

    count = len(canonical)
    expected = expected_count or json.loads(
        (candidate_dir / "candidate_manifest.json").read_text(encoding="utf-8")
    )["sample_count"]
    if count != expected:
        raise AssertionError(f"canonical count {count} != expected {expected}")
    if not all(len(rows) == count for rows in (messages, sft, docs, metadata)):
        raise AssertionError("candidate outputs do not have equal row counts")

    ids = []
    exact_docs = set()
    invalid_reasons: dict[str, int] = {}
    for index, (record, message_row, sft_row, doc_row, metadata_row) in enumerate(
        zip(canonical, messages, sft, docs, metadata)
    ):
        sample_id = record.get("sample_id")
        ids.append(sample_id)
        if sample_id != message_row.get("sample_id") or sample_id != doc_row.get("sample_id"):
            raise AssertionError(f"sample_id misalignment at row {index}")
        if record.get("source") != message_row.get("source") or record.get("source") != doc_row.get("source"):
            raise AssertionError(f"source misalignment at row {index}")
        if record.get("messages") != message_row.get("messages"):
            raise AssertionError(f"candidate_messages misalignment at row {index}")
        if sft_row.get("messages") != record.get("messages"):
            raise AssertionError(f"candidate_sft misalignment at row {index}")
        if doc_row.get("doc") != record.get("doc"):
            raise AssertionError(f"candidate_doc misalignment at row {index}")
        if metadata_row.get("row_index") != index:
            raise AssertionError(f"metadata row_index misalignment at row {index}")
        valid, reason = validate_messages(record.get("messages"))
        if not valid:
            invalid_reasons[reason or "unknown"] = invalid_reasons.get(reason or "unknown", 0) + 1
        built_doc = build_doc(record["messages"])
        if built_doc != record.get("doc"):
            raise AssertionError(f"doc mismatch at row {index}")
        if record["doc"] in exact_docs:
            raise AssertionError(f"exact duplicate doc at row {index}")
        exact_docs.add(record["doc"])

    duplicate_ids = len(ids) - len(set(ids))
    if duplicate_ids:
        raise AssertionError(f"duplicate sample IDs: {duplicate_ids}")
    if invalid_reasons:
        raise AssertionError(f"invalid messages: {invalid_reasons}")
    return {
        "candidate_dir": str(candidate_dir),
        "sample_count": count,
        "unique_sample_ids": len(set(ids)),
        "exact_duplicate_docs": 0,
        "aligned_outputs": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit SCS candidate pool outputs.")
    parser.add_argument("candidate_dir", type=Path)
    parser.add_argument("--expected-count", type=int, default=None)
    args = parser.parse_args()
    result = audit(args.candidate_dir.resolve(), args.expected_count)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

