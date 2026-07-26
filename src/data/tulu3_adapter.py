"""Adapter for allenai/tulu-3-sft-mixture rows."""

from __future__ import annotations

from typing import Any, Iterable

from tqdm.auto import tqdm

from .normalize import normalize_messages


def is_excluded_tulu_source(source: str) -> bool:
    lowered = source.lower()
    return "oasst" in lowered or "openassistant" in lowered


def adapt_tulu_row(row: dict[str, Any], row_index: int) -> dict[str, Any]:
    source = row.get("source")
    if not isinstance(source, str) or not source.strip():
        raise ValueError("Tulu row has no source")
    source = source.strip()
    if is_excluded_tulu_source(source):
        raise ValueError("excluded_tulu_oasst_source")

    original_id = row.get("id", row_index)
    messages = normalize_messages(row, record_index=row_index)
    metadata = {
        "source_dataset": "tulu3",
        "source_subset": source,
        "language": row.get("language", "en"),
        "original_id": str(original_id),
        "source_row_index": row_index,
    }
    return {
        "sample_id": f"tulu3::{source}::{original_id}",
        "source": source,
        "messages": messages,
        "metadata": metadata,
    }


def adapt_tulu(rows: Iterable[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    records = []
    removed: dict[str, int] = {}
    for row_index, row in enumerate(tqdm(rows, desc="Adapt Tulu records", unit="row")):
        try:
            records.append(adapt_tulu_row(row, row_index))
        except ValueError as exc:
            reason = str(exc)
            removed[reason] = removed.get(reason, 0) + 1
    return records, removed
