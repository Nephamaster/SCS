"""Small, dependency-light loaders for local dataset snapshots.

The server layout described by the pipeline uses local directories or symlinks.
This module intentionally supports JSON/JSONL first and only imports Hugging
Face Datasets when a parquet or ``load_from_disk`` snapshot is encountered.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any, Iterable, Iterator

from tqdm.auto import tqdm


JSON_SUFFIXES = {".json", ".jsonl", ".json.gz", ".jsonl.gz"}


def _open_text(path: Path):
    if path.name.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def _iter_json_file(path: Path) -> Iterator[dict[str, Any]]:
    with _open_text(path) as handle:
        if path.name.endswith(".jsonl") or path.name.endswith(".jsonl.gz"):
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"{path}:{line_number} is not a JSON object")
                yield value
            return

        value = json.load(handle)
        if isinstance(value, list):
            for index, item in enumerate(value):
                if not isinstance(item, dict):
                    raise ValueError(f"{path}[{index}] is not a JSON object")
                yield item
        elif isinstance(value, dict):
            # A few dataset snapshots wrap rows in a top-level ``data`` key.
            rows = value.get("data")
            if isinstance(rows, list):
                for index, item in enumerate(rows):
                    if not isinstance(item, dict):
                        raise ValueError(f"{path}.data[{index}] is not an object")
                    yield item
            else:
                yield value
        else:
            raise ValueError(f"Unsupported JSON root in {path}: {type(value).__name__}")


def _iter_parquet(path: Path, *, split_name: str = "train") -> Iterator[dict[str, Any]]:
    try:
        from datasets import load_dataset, load_from_disk
    except ImportError as exc:  # pragma: no cover - depends on server env
        raise RuntimeError(
            f"Reading parquet dataset {path} requires the 'datasets' package."
        ) from exc

    if path.is_dir() and (path / "dataset_info.json").exists():
        dataset = load_from_disk(str(path))
        if isinstance(dataset, dict):
            dataset = dataset.get(split_name) or next(iter(dataset.values()))
    else:
        dataset = load_dataset(
            "parquet",
            data_files={split_name: str(path)},
            split=split_name,
        )
    for row in dataset:
        yield dict(row)


def _files_for_path(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"Dataset path does not exist: {path}")

    files = []
    for candidate in path.rglob("*"):
        if not candidate.is_file():
            continue
        name = candidate.name.lower()
        if name.endswith(tuple(JSON_SUFFIXES)) or name.endswith(".parquet"):
            files.append(candidate)
    return sorted(files)


def iter_records(
    path: str | Path,
    *,
    relative_paths: Iterable[str | Path] | None = None,
    split_name: str = "train",
    description: str | None = None,
) -> Iterator[dict[str, Any]]:
    """Yield local dataset rows in deterministic file order."""

    resolved = Path(path).expanduser()

    if relative_paths is not None:
        if not resolved.is_dir():
            raise FileNotFoundError(f"Dataset root does not exist: {resolved}")
        expected_files = [resolved / Path(relative) for relative in relative_paths]
        missing = [str(file_path) for file_path in expected_files if not file_path.is_file()]
        if missing:
            raise FileNotFoundError(
                "Expected dataset files are missing:\n" + "\n".join(missing)
            )
        for file_path in tqdm(
            expected_files,
            desc=description or f"Loading {resolved.name}",
            unit="file",
        ):
            if file_path.name.lower().endswith(".parquet"):
                yield from _iter_parquet(file_path, split_name=split_name)
            else:
                yield from _iter_json_file(file_path)
        return

    if not resolved.exists():
        # Permit a Hugging Face dataset id when the server has network/cache
        # access. Local paths remain the documented and preferred mode.
        if isinstance(path, str) and "/" in path and not path.startswith((".", "/")):
            try:
                from datasets import load_dataset
            except ImportError as exc:  # pragma: no cover - server-only path
                raise FileNotFoundError(
                    f"Dataset path does not exist and loading Hub dataset {path!r} "
                    "requires the 'datasets' package."
                ) from exc
            dataset = load_dataset(path, split=split_name)
            for row in dataset:
                yield dict(row)
            return
        raise FileNotFoundError(f"Dataset path does not exist: {resolved}")

    if resolved.is_dir() and (resolved / "dataset_info.json").exists():
            yield from _iter_parquet(resolved, split_name=split_name)
            return

    files = _files_for_path(resolved)
    if not files:
        raise FileNotFoundError(f"No JSON/JSONL/parquet files found under {resolved}")
    for file_path in files:
        if file_path.name.lower().endswith(".parquet"):
            yield from _iter_parquet(file_path, split_name=split_name)
        else:
            yield from _iter_json_file(file_path)


def read_records(
    path: str | Path,
    *,
    relative_paths: Iterable[str | Path] | None = None,
    split_name: str = "train",
    description: str | None = None,
) -> list[dict[str, Any]]:
    return list(
        iter_records(
            path,
            relative_paths=relative_paths,
            split_name=split_name,
            description=description,
        )
    )
