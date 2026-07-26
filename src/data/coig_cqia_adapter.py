"""Adapter for m-a-p/COIG-CQIA rows."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable

from tqdm.auto import tqdm

from .normalize import clean_content


def _stable_hash(instruction: str, input_text: str, output: str) -> str:
    payload = json.dumps(
        {"instruction": instruction, "input": input_text, "output": output},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def adapt_coig_row(row: dict[str, Any], row_index: int) -> dict[str, Any]:
    instruction = clean_content(row.get("instruction"))
    input_value = row.get("input", "")
    if input_value is None:
        input_value = ""
    input_text = clean_content(input_value)
    output = clean_content(row.get("output"))
    if not instruction:
        raise ValueError("empty_instruction")
    if not output:
        raise ValueError("empty_output")

    user_content = instruction
    if input_text:
        user_content += "\n\n" + input_text

    config = str(row.get("config", row.get("subset", "default")))
    row_id = row.get("id", row.get("idx", row.get("row_id", row_index)))
    digest = _stable_hash(instruction, input_text, output)
    metadata = {
        "source_dataset": "coig_cqia",
        "source_config": config,
        "source_row_id": str(row_id),
        "language": row.get("language", "zh"),
        "task_type": row.get("task_type"),
        "domain": row.get("domain"),
        "answer_from": row.get("answer_from"),
        "human_verified": row.get("human_verified", False),
        "copyright": row.get("copyright"),
        "source_row_index": row_index,
    }
    return {
        "sample_id": f"coig_cqia::{config}::{row_id}::{digest}",
        "source": "coig_cqia",
        "messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": output},
        ],
        "metadata": metadata,
    }


def adapt_coig(rows: Iterable[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    records = []
    removed: dict[str, int] = {}
    for row_index, row in enumerate(tqdm(rows, desc="Adapt COIG-CQIA records", unit="row")):
        try:
            records.append(adapt_coig_row(row, row_index))
        except (TypeError, ValueError) as exc:
            reason = str(exc)
            removed[reason] = removed.get(reason, 0) + 1
    return records, removed
