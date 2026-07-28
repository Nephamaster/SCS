"""Calculate SCS for candidate-indexed selection datasets."""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from tqdm.auto import tqdm

from .entropy import cal_class_prob, cal_cohesion_weights


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT_DIR / "data" / "sft" / "random"
DEFAULT_CANDIDATE_DOC = ROOT_DIR / "data" / "candidate" / "v1" / "candidate_doc.jsonl"
DEFAULT_EMBEDDING_NPZ = (
    ROOT_DIR / "output" / "feature" / "candidate_embeddings_llama.npz"
)
DEFAULT_LOGPROB_NPZ = (
    ROOT_DIR / "output" / "feature" / "candidate_logprob_llama.npz"
)
DEFAULT_OUTPUT_DIR = ROOT_DIR / "output" / "scs"


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT_DIR / path


def load_jsonl(path: Path, description: str) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(
            tqdm(handle, desc=description, unit="row"), start=1
        ):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number} is not an object.")
            records.append(record)
    if not records:
        raise ValueError(f"Input file is empty: {path}")
    return records


def input_files(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        files = sorted(input_path.glob("*.jsonl"))
        if not files:
            raise FileNotFoundError(f"No JSONL datasets found in: {input_path}")
        return files
    raise FileNotFoundError(f"Input path not found: {input_path}")


def load_features(
    embedding_path: Path,
    logprob_path: Path,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    if not embedding_path.exists():
        raise FileNotFoundError(f"Embedding NPZ not found: {embedding_path}")
    if not logprob_path.exists():
        raise FileNotFoundError(f"Log-probability NPZ not found: {logprob_path}")

    with np.load(embedding_path, allow_pickle=False) as data:
        if "embeddings" not in data or "sample_ids" not in data:
            raise ValueError("Embedding NPZ must contain embeddings and sample_ids.")
        embeddings = np.asarray(data["embeddings"], dtype=np.float32)
        embedding_ids = np.asarray(data["sample_ids"]).astype(str).tolist()

    with np.load(logprob_path, allow_pickle=False) as data:
        if "ln_probability" not in data or "sample_ids" not in data:
            raise ValueError(
                "Log-probability NPZ must contain ln_probability and sample_ids."
            )
        ln_probabilities = np.asarray(data["ln_probability"], dtype=np.float32)
        logprob_ids = np.asarray(data["sample_ids"]).astype(str).tolist()

    if embeddings.ndim != 2 or len(embeddings) != len(embedding_ids):
        raise ValueError("Embedding NPZ shape does not match sample_ids.")
    if ln_probabilities.ndim != 1 or len(ln_probabilities) != len(logprob_ids):
        raise ValueError("Log-probability NPZ shape does not match sample_ids.")
    if embedding_ids != logprob_ids:
        raise ValueError("Embedding and log-probability sample_ids are not aligned.")
    if not np.isfinite(embeddings).all():
        raise ValueError("Embedding NPZ contains NaN or infinity values.")
    if not np.isfinite(ln_probabilities).all():
        raise ValueError("Log-probability NPZ contains NaN or infinity values.")
    return embeddings, ln_probabilities, embedding_ids


def resolve_indices(
    records: list[dict[str, Any]],
    sample_id_to_index: dict[str, int],
    sample_ids: list[str],
) -> list[int]:
    indices = []
    seen = set()
    for record_index, record in enumerate(records):
        candidate_index = record.get("candidate_index")
        sample_id = record.get("sample_id")
        if candidate_index is None:
            if not isinstance(sample_id, str) or sample_id not in sample_id_to_index:
                raise ValueError(
                    f"Record {record_index} needs candidate_index or a known sample_id."
                )
            candidate_index = sample_id_to_index[sample_id]
        if isinstance(candidate_index, bool) or not isinstance(candidate_index, int):
            raise ValueError(f"Record {record_index} has an invalid candidate_index.")
        if not 0 <= candidate_index < len(sample_ids):
            raise ValueError(
                f"Record {record_index} candidate_index is out of range: {candidate_index}."
            )
        if isinstance(sample_id, str) and sample_ids[candidate_index] != sample_id:
            raise ValueError(
                f"Record {record_index} candidate_index/sample_id mismatch."
            )
        if candidate_index in seen:
            raise ValueError(f"Duplicate candidate_index in selection: {candidate_index}")
        seen.add(candidate_index)
        indices.append(candidate_index)
    return indices


def load_selected_docs(
    candidate_doc_path: Path,
    selected_indices: set[int],
    sample_ids: list[str],
) -> dict[int, str]:
    """Read only selected DOC rows from the full candidate DOC file."""
    if not candidate_doc_path.exists():
        raise FileNotFoundError(f"Candidate DOC file not found: {candidate_doc_path}")

    docs: dict[int, str] = {}
    candidate_index = 0
    with candidate_doc_path.open("r", encoding="utf-8") as handle:
        for line in tqdm(handle, desc="Load candidate DOC", unit="row"):
            if not line.strip():
                continue
            row_index = candidate_index
            candidate_index += 1
            if row_index not in selected_indices:
                continue
            record = json.loads(line)
            if not isinstance(record, dict) or not isinstance(record.get("doc"), str):
                raise ValueError(f"Candidate DOC row {row_index} has no string doc.")
            sample_id = record.get("sample_id")
            if isinstance(sample_id, str) and sample_id != sample_ids[row_index]:
                raise ValueError(f"Candidate DOC row {row_index} is misaligned.")
            docs[row_index] = record["doc"]

    missing = sorted(selected_indices - docs.keys())
    if missing:
        raise ValueError(f"Candidate DOC is missing selected indices: {missing[:10]}")
    return docs


def build_semantic_ids(
    embeddings: np.ndarray,
    seed: int,
    min_cluster_size: int,
    n_neighbors: int,
    n_components: int,
) -> np.ndarray:
    if len(embeddings) < min_cluster_size:
        raise ValueError(
            f"Dataset has {len(embeddings)} rows, fewer than min_cluster_size "
            f"({min_cluster_size})."
        )
    import hdbscan
    from umap import UMAP

    reducer = UMAP(
        n_neighbors=min(n_neighbors, len(embeddings) - 1),
        n_components=min(n_components, embeddings.shape[1], len(embeddings) - 1),
        metric="cosine",
        random_state=seed,
    )
    reduced = reducer.fit_transform(embeddings)
    labels = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        metric="euclidean",
        cluster_selection_method="eom",
    ).fit_predict(reduced)

    cluster_count = len(set(labels)) - (1 if -1 in labels else 0)
    return np.asarray(
        [int(label) if label != -1 else cluster_count for label in labels],
        dtype=np.int64,
    )


def calculate_scs(
    dataset_name: str,
    embeddings: np.ndarray,
    ln_probabilities: np.ndarray,
    semantic_ids: np.ndarray,
    token_counts: np.ndarray,
) -> dict[str, Any]:
    sem_ids = semantic_ids.tolist()
    cohesion_weights = cal_cohesion_weights(sem_ids, embeddings)
    class_cohe_probs, class_gen_probs, class_probs = cal_class_prob(
        sem_ids,
        ln_probabilities.tolist(),
        cohesion_weights,
    )
    sce = float(-np.sum(class_cohe_probs * np.log(class_cohe_probs)))
    sce_wo_cohesion = float(-np.sum(class_gen_probs * np.log(class_gen_probs)))
    sce_wo_generator = float(-np.sum(class_probs * np.log(class_probs)))

    class_content = {}
    for class_id in range(len(class_probs)):
        class_indices = [index for index, sem_id in enumerate(sem_ids) if sem_id == class_id]
        class_content[f"class_{class_id}"] = {
            "percentage": float(class_probs[class_id]),
            "ln_probability": float(class_gen_probs[class_id]),
            "cohesion": float(class_cohe_probs[class_id]),
            "avg_token_num": float(np.mean(token_counts[class_indices])),
        }

    return {
        "dataset_name": dataset_name,
        "SCE": sce,
        "SCE_wo_cohesion": sce_wo_cohesion,
        "SCE_wo_cohesion_wo_generator": sce_wo_generator,
        "class_num": len(class_probs),
        "class_content": class_content,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calculate SCS for candidate-indexed JSONL datasets."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--candidate-doc",
        "--candidate_doc",
        type=Path,
        default=DEFAULT_CANDIDATE_DOC,
    )
    parser.add_argument(
        "--embedding-npz",
        "--embedding_npz",
        type=Path,
        default=DEFAULT_EMBEDDING_NPZ,
    )
    parser.add_argument(
        "--logprob-npz",
        "--logprob_npz",
        type=Path,
        default=DEFAULT_LOGPROB_NPZ,
    )
    parser.add_argument(
        "--tokenizer",
        default="FacebookAI/xlm-roberta-large",
        help="Tokenizer used for avg_token_num.",
    )
    parser.add_argument(
        "--output-dir",
        "--output_dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument("--seed", type=int, default=46)
    parser.add_argument(
        "--min-cluster-size", "--min_cluster_size", type=int, default=30
    )
    parser.add_argument("--n-neighbors", "--n_neighbors", type=int, default=15)
    parser.add_argument("--n-components", "--n_components", type=int, default=32)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = resolve_path(args.input)
    candidate_doc_path = resolve_path(args.candidate_doc)
    embedding_path = resolve_path(args.embedding_npz)
    logprob_path = resolve_path(args.logprob_npz)
    output_dir = resolve_path(args.output_dir)

    files = input_files(input_path)
    selections = []
    all_indices = set()
    embeddings, ln_probabilities, sample_ids = load_features(
        embedding_path,
        logprob_path,
    )
    sample_id_to_index = {
        sample_id: index for index, sample_id in enumerate(sample_ids)
    }
    for path in files:
        records = load_jsonl(path, f"Load {path.name}")
        indices = resolve_indices(records, sample_id_to_index, sample_ids)
        selections.append((path, indices))
        all_indices.update(indices)

    docs = load_selected_docs(candidate_doc_path, all_indices, sample_ids)
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Calculating avg_token_num requires transformers."
        ) from exc
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        clean_up_tokenization_spaces=False,
        model_max_length=4096,
    )
    token_counts = {
        index: len(tokenizer.tokenize(docs[index]))
        for index in tqdm(sorted(all_indices), desc="Count DOC tokens", unit="row")
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    for path, indices in selections:
        output_path = output_dir / f"{path.stem}_SCS.json"
        if output_path.exists() and not args.overwrite:
            print(f"Skip existing SCS result: {output_path}")
            continue

        index_array = np.asarray(indices, dtype=np.int64)
        selected_embeddings = embeddings[index_array]
        selected_probs = ln_probabilities[index_array]
        selected_tokens = np.asarray([token_counts[index] for index in indices])
        semantic_ids = build_semantic_ids(
            selected_embeddings,
            args.seed,
            args.min_cluster_size,
            args.n_neighbors,
            args.n_components,
        )
        result = calculate_scs(
            path.stem,
            selected_embeddings,
            selected_probs,
            semantic_ids,
            selected_tokens,
        )
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        print(f"Saved SCS: {output_path}")


if __name__ == "__main__":
    main()
