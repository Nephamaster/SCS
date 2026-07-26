"""Deterministic OASST2 validation tree reconstruction."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable

from tqdm.auto import tqdm

from .normalize import clean_content


def _keep_node(row: dict[str, Any]) -> bool:
    return (
        row.get("deleted") is False
        and row.get("review_result") is not False
        and row.get("tree_state") == "ready_for_export"
    )


def _assistant_key(row: dict[str, Any]) -> tuple[int, int, str]:
    rank = row.get("rank")
    if rank == 0:
        return (0, 0, str(row.get("message_id", "")))
    if isinstance(rank, (int, float)):
        return (1, int(rank), str(row.get("message_id", "")))
    return (2, 0, str(row.get("message_id", "")))


def _choose_child(children: list[dict[str, Any]], role: str) -> dict[str, Any] | None:
    candidates = [row for row in children if row.get("role") == role]
    if not candidates:
        return None
    if role == "assistant":
        return sorted(candidates, key=_assistant_key)[0]
    return sorted(candidates, key=lambda row: str(row.get("message_id", "")))[0]


def build_oasst2_validation(rows: Iterable[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    trees: dict[str, list[dict[str, Any]]] = defaultdict(list)
    removed: dict[str, int] = {}
    for row in tqdm(rows, desc="Filter OASST2 validation nodes", unit="row"):
        if not _keep_node(row):
            removed["oasst_node_filter"] = removed.get("oasst_node_filter", 0) + 1
            continue
        tree_id = row.get("message_tree_id")
        message_id = row.get("message_id")
        if not tree_id or not message_id or not isinstance(row.get("text"), str):
            removed["oasst_invalid_node"] = removed.get("oasst_invalid_node", 0) + 1
            continue
        trees[str(tree_id)].append(row)

    outputs = []
    for tree_id in tqdm(
        sorted(trees),
        desc="Rebuild OASST2 validation trees",
        unit="tree",
    ):
        nodes = trees[tree_id]
        by_id = {str(row["message_id"]): row for row in nodes}
        children: dict[str | None, list[dict[str, Any]]] = defaultdict(list)
        for row in nodes:
            parent = row.get("parent_id")
            children[None if parent is None else str(parent)].append(row)

        roots = [row for row in children[None] if row.get("role") == "prompter"]
        if not roots:
            removed["oasst_no_prompter_root"] = removed.get("oasst_no_prompter_root", 0) + 1
            continue
        current = sorted(roots, key=lambda row: str(row["message_id"]))[0]
        path = [current]
        while True:
            current_role = current.get("role")
            expected = "assistant" if current_role == "prompter" else "prompter"
            next_node = _choose_child(
                children.get(str(current["message_id"]), []), expected
            )
            if next_node is None:
                break
            path.append(next_node)
            current = next_node

        last_assistant = max(
            (index for index, row in enumerate(path) if row.get("role") == "assistant"),
            default=-1,
        )
        if last_assistant < 1:
            removed["oasst_no_assistant_path"] = removed.get("oasst_no_assistant_path", 0) + 1
            continue
        path = path[: last_assistant + 1]
        messages = []
        valid = True
        for row in path:
            role = "user" if row["role"] == "prompter" else "assistant"
            content = clean_content(row["text"])
            if not content:
                valid = False
                break
            messages.append({"role": role, "content": content})
        if not valid or messages[-1]["role"] != "assistant":
            removed["oasst_invalid_path"] = removed.get("oasst_invalid_path", 0) + 1
            continue

        leaf_id = str(path[-1]["message_id"])
        outputs.append(
            {
                "sample_id": f"oasst2::{tree_id}::{leaf_id}",
                "source": "oasst2_validation",
                "messages": messages,
                "metadata": {
                    "message_tree_id": tree_id,
                    "leaf_message_id": leaf_id,
                    "language": path[0].get("lang", ""),
                },
            }
        )
    return outputs, removed
