"""Data-Juicer filter for canonical SFT message structure."""

from __future__ import annotations

try:  # Data-Juicer is installed on the server, not in the local test env.
    from data_juicer.ops.base_op import OPERATORS, Filter
except ImportError:  # pragma: no cover
    OPERATORS = None

    class Filter:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            pass

try:
    from src.data.normalize import validate_messages
except ModuleNotFoundError:  # loaded by Data-Juicer from the src directory
    from data.normalize import validate_messages


if OPERATORS is not None:

    @OPERATORS.register_module("scs_messages_schema_filter")
    class SCSMessagesSchemaFilter(Filter):
        """Keep only records with valid assistant-supervised messages."""

        def process_single(self, sample):
            valid, _ = validate_messages(sample.get("messages"))
            return valid

else:

    class SCSMessagesSchemaFilter(Filter):  # type: ignore[no-redef]
        def process_single(self, sample):
            valid, _ = validate_messages(sample.get("messages"))
            return valid
