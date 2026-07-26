"""Deterministic exact and MinHash deduplication helpers.

The implementation mirrors the Data-Juicer recipe parameters while keeping
the source-priority rule explicit. The final role-aware ``doc`` is never
written for the pre-candidate population; callers pass an internal text view.
"""

from __future__ import annotations

import hashlib
import random
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable, Iterable

from tqdm.auto import tqdm


MERSENNE_PRIME = (1 << 61) - 1
MAX_HASH = (1 << 32) - 1


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _base_hash(value: str) -> int:
    return int.from_bytes(
        hashlib.sha1(value.encode("utf-8")).digest()[:8], "little"
    ) & MAX_HASH


def character_ngrams(text: str, window_size: int = 5) -> frozenset[str]:
    if len(text) < window_size:
        return frozenset()
    return frozenset(text[index : index + window_size] for index in range(len(text) - window_size + 1))


@lru_cache(maxsize=4)
def _load_sentencepiece(tokenizer_model: str):
    try:
        import sentencepiece as spm
    except ImportError as exc:  # pragma: no cover - server dependency
        raise RuntimeError(
            "sentencepiece MinHash requires the 'sentencepiece' package"
        ) from exc
    return spm.SentencePieceProcessor(model_file=tokenizer_model)


@lru_cache(maxsize=4)
def _load_huggingface_tokenizer(tokenizer_source: str):
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover - server dependency
        raise RuntimeError(
            "Hugging Face tokenizer MinHash requires the 'transformers' package"
        ) from exc
    source = Path(tokenizer_source)
    return AutoTokenizer.from_pretrained(
        str(source),
        use_fast=True,
        local_files_only=source.exists(),
    )


def sentencepiece_ngrams(
    text: str,
    tokenizer_model: str | None,
    window_size: int = 5,
) -> frozenset[str]:
    """Build token shingles from SentencePiece or a Hugging Face tokenizer.

    The historical pipeline called this stage ``sentencepiece``. Llama
    checkpoints expose a standalone ``tokenizer.model`` while Qwen3 exposes a
    Hugging Face tokenizer directory, so both local layouts are accepted.
    This is only for MinHash shingling; the candidate build performs no length
    or token-count audit.
    """
    if not tokenizer_model:
        raise ValueError("MinHash requires --minhash-tokenizer-model")

    path = Path(tokenizer_model).expanduser()
    if path.is_file() and path.suffix.lower() == ".model":
        tokens = _load_sentencepiece(str(path)).encode(text, out_type=str)
    else:
        source = path.parent if path.is_file() else path
        tokenizer = _load_huggingface_tokenizer(str(source))
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        tokens = tokenizer.convert_ids_to_tokens(token_ids)
    if len(tokens) < window_size:
        return frozenset()
    return frozenset(
        "".join(tokens[index : index + window_size])
        for index in range(len(tokens) - window_size + 1)
    )


class MinHashFamily:
    """A deterministic MinHash family compatible with 256 permutations."""

    def __init__(self, num_permutations: int = 256, seed: int = 42):
        self.num_permutations = num_permutations
        try:
            import numpy as np
        except ImportError:  # pragma: no cover - numpy is a server dependency
            rng = random.Random(seed)
            self.perm_a = [
                rng.randrange(1, MERSENNE_PRIME) for _ in range(num_permutations)
            ]
            self.perm_b = [
                rng.randrange(0, MERSENNE_PRIME) for _ in range(num_permutations)
            ]
        else:
            # Match Data-Juicer's fixed RandomState(42)-style permutation
            # generation while keeping Python integers for the pure-Python LSH.
            rng = np.random.RandomState(seed=seed)
            self.perm_a = [
                int(rng.randint(1, MERSENNE_PRIME, dtype=np.uint64))
                for _ in range(num_permutations)
            ]
            self.perm_b = [
                int(rng.randint(0, MERSENNE_PRIME, dtype=np.uint64))
                for _ in range(num_permutations)
            ]

    def signature(self, shingles: Iterable[str]) -> tuple[int, ...]:
        hashes = [_base_hash(shingle) for shingle in set(shingles)]
        if not hashes:
            return tuple([MERSENNE_PRIME] * self.num_permutations)
        return tuple(
            min(((a * value + b) % MERSENNE_PRIME) & MAX_HASH for value in hashes)
            for a, b in zip(self.perm_a, self.perm_b)
        )


@dataclass(frozen=True)
class NearDuplicateMatch:
    kept_index: int
    duplicate_index: int
    similarity: float


class MinHashLSHIndex:
    """Index signatures and verify candidate pairs with exact Jaccard."""

    def __init__(self, family: MinHashFamily, bands: int = 32):
        if family.num_permutations % bands:
            raise ValueError("num_permutations must be divisible by bands")
        self.family = family
        self.bands = bands
        self.rows_per_band = family.num_permutations // bands
        self.buckets: dict[tuple[int, tuple[int, ...]], list[int]] = defaultdict(list)
        self.shingles: list[frozenset[str]] = []
        self.signatures: list[tuple[int, ...]] = []

    def add(self, shingles: frozenset[str]) -> int:
        index = len(self.shingles)
        signature = self.family.signature(shingles)
        self.shingles.append(shingles)
        self.signatures.append(signature)
        for band in range(self.bands):
            start = band * self.rows_per_band
            end = start + self.rows_per_band
            self.buckets[(band, signature[start:end])].append(index)
        return index

    def candidates(self, signature: tuple[int, ...]) -> set[int]:
        result: set[int] = set()
        for band in range(self.bands):
            start = band * self.rows_per_band
            end = start + self.rows_per_band
            result.update(self.buckets.get((band, signature[start:end]), []))
        return result


def jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    if not left and not right:
        return 1.0
    union = len(left | right)
    return len(left & right) / union if union else 0.0


def exact_deduplicate(
    records: list[dict],
    text_getter: Callable[[dict], str],
    priority_getter: Callable[[dict], tuple],
) -> tuple[list[dict], list[dict]]:
    ordered = sorted(
        enumerate(records), key=lambda item: (priority_getter(item[1]), item[1]["sample_id"])
    )
    seen: dict[str, dict] = {}
    kept: list[dict] = []
    removed: list[dict] = []
    for original_index, record in tqdm(
        ordered,
        desc="Exact document deduplication",
        unit="row",
    ):
        digest = sha256_text(text_getter(record))
        if digest in seen:
            removed.append(
                {
                    "sample_id": record["sample_id"],
                    "matched_sample_id": seen[digest]["sample_id"],
                    "match_type": "exact_doc",
                }
            )
            continue
        seen[digest] = record
        kept.append(record)
    return kept, removed


def minhash_deduplicate(
    records: list[dict],
    text_getter: Callable[[dict], str],
    priority_getter: Callable[[dict], tuple],
    shingle_getter: Callable[[str], frozenset[str]],
    *,
    num_permutations: int = 256,
    threshold: float = 0.80,
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    ordered = sorted(
        records,
        key=lambda record: (priority_getter(record), record["sample_id"]),
    )
    family = MinHashFamily(num_permutations=num_permutations, seed=seed)
    index = MinHashLSHIndex(family)
    kept: list[dict] = []
    removed: list[dict] = []

    for record in tqdm(ordered, desc="MinHash document deduplication", unit="row"):
        shingles = shingle_getter(text_getter(record))
        signature = family.signature(shingles)
        match = None
        for kept_index in index.candidates(signature):
            similarity = jaccard(shingles, index.shingles[kept_index])
            if similarity >= threshold:
                candidate = (similarity, index.shingles[kept_index], kept_index)
                if match is None or candidate[0] > match[0]:
                    match = candidate
        if match is not None:
            kept_record = kept[match[2]]
            removed.append(
                {
                    "sample_id": record["sample_id"],
                    "matched_sample_id": kept_record["sample_id"],
                    "match_type": "minhash_doc",
                    "similarity": round(float(match[0]), 6),
                }
            )
            continue
        index.add(shingles)
        kept.append(record)
    return kept, removed


class PromptDecontaminationIndex:
    """Exact and character-5-gram approximate prompt matching against dev."""

    def __init__(self, dev_records: list[dict], *, threshold: float = 0.80):
        self.threshold = threshold
        self.family = MinHashFamily(256, seed=42)
        self.lsh = MinHashLSHIndex(self.family)
        self.prompt_sets: list[frozenset[str]] = []
        self.dev_records = dev_records
        self.exact_doc: dict[str, list[int]] = defaultdict(list)
        self.exact_prompt: dict[str, list[int]] = defaultdict(list)
        for index, record in enumerate(
            tqdm(dev_records, desc="Index OASST2 validation prompts", unit="row")
        ):
            doc_text = record["_dedup_text"]
            prompt = record["_first_user_prompt"]
            self.exact_doc[sha256_text(doc_text)].append(index)
            self.exact_prompt[sha256_text(prompt)].append(index)
            prompt_set = character_ngrams(prompt, 5)
            self.prompt_sets.append(prompt_set)
            self.lsh.add(prompt_set)

    def match(self, record: dict) -> dict | None:
        doc_digest = sha256_text(record["_dedup_text"])
        if doc_digest in self.exact_doc:
            index = min(self.exact_doc[doc_digest], key=lambda i: self.dev_records[i]["sample_id"])
            return {
                "matched_dev_id": self.dev_records[index]["sample_id"],
                "match_type": "exact_doc",
            }

        prompt = record["_first_user_prompt"]
        prompt_digest = sha256_text(prompt)
        if prompt_digest in self.exact_prompt:
            index = min(self.exact_prompt[prompt_digest], key=lambda i: self.dev_records[i]["sample_id"])
            return {
                "matched_dev_id": self.dev_records[index]["sample_id"],
                "match_type": "exact_prompt",
            }

        prompt_set = character_ngrams(prompt, 5)
        if not prompt_set:
            return None
        signature = self.family.signature(prompt_set)
        best: tuple[float, str] | None = None
        for index in self.lsh.candidates(signature):
            similarity = jaccard(prompt_set, self.prompt_sets[index])
            if similarity < self.threshold:
                continue
            candidate = (similarity, self.dev_records[index]["sample_id"])
            if best is None or candidate[0] > best[0] or (
                candidate[0] == best[0] and candidate[1] < best[1]
            ):
                best = candidate
        if best is None:
            return None
        return {
            "matched_dev_id": best[1],
            "match_type": "prompt_minhash",
            "similarity": round(float(best[0]), 6),
        }
