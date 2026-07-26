"""Canonical message normalization and validation."""

from __future__ import annotations

import re
import unicodedata
from typing import Any


ROLE_ALIASES = {
    "human": "user",
    "prompter": "user",
    "user": "user",
    "gpt": "assistant",
    "bot": "assistant",
    "model": "assistant",
    "assistant": "assistant",
    "system": "system",
    "tool": "tool",
    "tool_response": "tool",
    "tool_call": "tool_call",
}
SUPPORTED_ROLES = {"system", "user", "assistant", "tool", "tool_call"}
CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def clean_content(value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError("message content must be a string")
    value = unicodedata.normalize("NFKC", value)
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    value = CONTROL_CHARS.sub("", value)
    return value.strip()


def normalize_role(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("message role must be a non-empty string")
    role = ROLE_ALIASES.get(value.strip().lower(), value.strip().lower())
    if role not in SUPPORTED_ROLES:
        raise ValueError(f"unsupported message role: {value!r}")
    return role


def normalize_messages(
    record: dict[str, Any],
    *,
    record_index: int,
    allow_conversations: bool = True,
) -> list[dict[str, str]]:
    if isinstance(record.get("messages"), list):
        turns = record["messages"]
        role_key, content_key = "role", "content"
    elif allow_conversations and isinstance(record.get("conversations"), list):
        turns = record["conversations"]
        role_key, content_key = "from", "value"
    else:
        raise ValueError(f"record {record_index} has no messages/conversations list")

    messages: list[dict[str, str]] = []
    for turn_index, turn in enumerate(turns):
        if not isinstance(turn, dict):
            raise ValueError(f"record {record_index} turn {turn_index} is not an object")
        try:
            role = normalize_role(turn.get(role_key))
            content = clean_content(turn.get(content_key))
        except ValueError as exc:
            raise ValueError(
                f"record {record_index} turn {turn_index}: {exc}"
            ) from exc
        messages.append({"role": role, "content": content})
    return messages


def validate_messages(messages: Any) -> tuple[bool, str | None]:
    if not isinstance(messages, list) or not messages:
        return False, "empty_messages"

    normalized_roles: list[str] = []
    for message in messages:
        if not isinstance(message, dict):
            return False, "message_not_object"
        role = message.get("role")
        content = message.get("content")
        if role not in SUPPORTED_ROLES:
            return False, "invalid_role"
        if not isinstance(content, str):
            return False, "content_not_string"
        normalized_roles.append(role)

    first_non_system = next(
        (role for role in normalized_roles if role != "system"), None
    )
    if first_non_system != "user":
        return False, "first_non_system_not_user"
    if normalized_roles[-1] != "assistant":
        return False, "last_role_not_assistant"

    # Tool messages are accepted as tool responses and may occur between a
    # tool call/assistant turn and the next assistant turn. Ordinary user and
    # assistant turns must still alternate strictly.
    last_dialog_role: str | None = None
    tool_intervened = False
    seen_assistant = False
    for role, message in zip(normalized_roles, messages):
        if role == "system":
            if last_dialog_role is not None:
                return False, "system_not_at_prefix"
            continue
        if role == "assistant":
            if not message.get("content", "").strip():
                return False, "empty_assistant"
            if last_dialog_role == "assistant" and not tool_intervened:
                return False, "assistant_not_alternating"
            seen_assistant = True
            last_dialog_role = "assistant"
            tool_intervened = False
            continue
        if role == "user":
            if last_dialog_role == "user" and not tool_intervened:
                return False, "user_not_alternating"
            last_dialog_role = "user"
            tool_intervened = False
            continue
        if role in {"tool", "tool_call"}:
            # These roles are intentionally not treated as supervised turns.
            # They do not break user/assistant alternation.
            tool_intervened = True
            continue

    if not seen_assistant:
        return False, "no_assistant"
    return True, None


def first_user_prompt(messages: list[dict[str, str]]) -> str:
    for message in messages:
        if message["role"] == "user":
            return message["content"]
    return ""


def match_normalize(text: str) -> str:
    """Normalize text for exact decontamination hashes.

    Internal whitespace is deliberately preserved. This keeps the matching
    rule auditable and avoids silently changing training content.
    """

    return clean_content(text)
