"""Role-aware document representation used only after candidate selection."""

from __future__ import annotations

from typing import Iterable


ROLE_LABELS = {
    "system": "SYSTEM",
    "user": "USER",
    "assistant": "ASSISTANT",
    "tool": "TOOL",
    "tool_call": "TOOL_CALL",
}


def build_doc(messages: Iterable[dict[str, str]]) -> str:
    parts = []
    for message in messages:
        role = message["role"]
        if role not in ROLE_LABELS:
            raise ValueError(f"cannot build doc for role {role!r}")
        parts.append(f"[{ROLE_LABELS[role]}]\n{message['content'].strip()}")
    return "\n\n<TURN_END>\n\n".join(parts)

