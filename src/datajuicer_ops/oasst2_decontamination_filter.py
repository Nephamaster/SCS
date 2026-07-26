"""Data-Juicer-compatible OASST2 exact hash decontamination filter."""

from __future__ import annotations

try:
    from data_juicer.ops.base_op import OPERATORS, Filter
except ImportError:  # pragma: no cover
    OPERATORS = None

    class Filter:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            pass

try:
    from src.data.dedup import sha256_text
except ModuleNotFoundError:  # loaded by Data-Juicer from the src directory
    from data.dedup import sha256_text


if OPERATORS is not None:

    @OPERATORS.register_module("scs_oasst2_decontamination_filter")
    class SCSOASST2DecontaminationFilter(Filter):
        def __init__(self, doc_hashes=None, prompt_hashes=None, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.doc_hashes = set(doc_hashes or [])
            self.prompt_hashes = set(prompt_hashes or [])

        def process_single(self, sample):
            return not (
                sha256_text(sample.get("dedup_text", "")) in self.doc_hashes
                or sha256_text(sample.get("first_user_prompt", "")) in self.prompt_hashes
            )

else:

    class SCSOASST2DecontaminationFilter(Filter):  # type: ignore[no-redef]
        def __init__(self, doc_hashes=None, prompt_hashes=None, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.doc_hashes = set(doc_hashes or [])
            self.prompt_hashes = set(prompt_hashes or [])

        def process_single(self, sample):
            return not (
                sha256_text(sample.get("dedup_text", "")) in self.doc_hashes
                or sha256_text(sample.get("first_user_prompt", "")) in self.prompt_hashes
            )
