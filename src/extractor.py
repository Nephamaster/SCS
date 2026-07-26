import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
from tqdm.auto import tqdm


def _post_json(
    request: Request,
    *,
    label: str,
    timeout: float,
    max_retries: int,
) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            with urlopen(request, timeout=timeout) as handle:
                response = json.loads(handle.read().decode("utf-8"))
            if "error" in response:
                raise RuntimeError(f"vLLM {label} error: {response['error']}")
            return response
        except HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            message = f"vLLM {label} HTTP {exc.code}: {body[:1000]}"
            if exc.code < 500 and exc.code not in {408, 429}:
                raise RuntimeError(message) from exc
            last_error = RuntimeError(message)
        except (URLError, TimeoutError, OSError) as exc:
            last_error = exc

        if attempt < max_retries:
            time.sleep(min(2**attempt, 30))

    raise RuntimeError(
        f"vLLM {label} request failed after {max_retries + 1} attempts"
    ) from last_error


def _as_embedding_matrix(
    vectors: Sequence[Sequence[float]],
    expected_rows: int,
    normalize: bool,
) -> np.ndarray:
    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] != expected_rows:
        raise RuntimeError(f"Invalid embedding matrix shape: {matrix.shape}")
    if not np.isfinite(matrix).all():
        raise RuntimeError("vLLM returned a non-finite embedding")
    if normalize:
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        matrix = matrix / np.clip(norms, 1e-12, None)
    return matrix


def _endpoint(base_url: str, path: str) -> str:
    base_url = base_url.rstrip("/")
    if base_url.endswith(path):
        return base_url
    if base_url.endswith("/v1"):
        return base_url + path
    return base_url + "/v1" + path


class VLLMEmbeddingExtractor:
    """Embedding client for vLLM's OpenAI-compatible API."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8000/v1",
        model: str | None = None,
        api_key: str | None = None,
        batch_size: int = 64,
        timeout: float = 600.0,
        max_retries: int = 5,
        normalize: bool = True,
        max_len: int = 1024,
        max_workers: int = 4,
    ):
        if not model:
            raise ValueError("vLLM embedding model name is required")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        if max_retries < 0:
            raise ValueError("max_retries cannot be negative")
        if max_len <= 0:
            raise ValueError("max_len must be positive")
        if max_workers <= 0:
            raise ValueError("max_workers must be positive")

        self.endpoint = _endpoint(base_url, "/embeddings")
        self.model = model
        self.api_key = api_key
        self.batch_size = batch_size
        self.timeout = timeout
        self.max_retries = max_retries
        self.normalize = normalize
        self.max_len = max_len
        self.max_workers = max_workers

    def _post_batch(self, sentences: Sequence[str]) -> np.ndarray:
        payload = {
            "model": self.model,
            "input": list(sentences),
            "encoding_format": "float",
            "truncate_prompt_tokens": self.max_len,
            "truncation_side": "right",
        }
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = Request(
            self.endpoint,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        response = _post_json(
            request,
            label="embeddings",
            timeout=self.timeout,
            max_retries=self.max_retries,
        )
        data = response.get("data")
        if not isinstance(data, list) or len(data) != len(sentences):
            raise RuntimeError(
                "vLLM embeddings response length does not match request: "
                f"requested={len(sentences)}, "
                f"returned={len(data) if isinstance(data, list) else 'invalid'}"
            )
        ordered = sorted(
            enumerate(data),
            key=lambda item: int(item[1].get("index", item[0])),
        )
        return _as_embedding_matrix(
            [item[1]["embedding"] for item in ordered],
            len(sentences),
            self.normalize,
        )

    def get_embeddings(
        self,
        sentences: Sequence[str],
        *,
        batch_size: int | None = None,
        show_progress: bool = True,
    ) -> np.ndarray:
        texts = list(sentences)
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        size = batch_size or self.batch_size
        if size <= 0:
            raise ValueError("batch_size must be positive")

        batches = [texts[start : start + size] for start in range(0, len(texts), size)]
        vectors = []
        with tqdm(
            total=len(texts),
            desc="vLLM API embeddings",
            unit="row",
            disable=not show_progress,
        ) as progress:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                for batch, vector in zip(batches, executor.map(self._post_batch, batches)):
                    vectors.append(vector)
                    progress.update(len(batch))
        return np.concatenate(vectors, axis=0)

    def get_embedding(self, sentence: str) -> np.ndarray:
        return self.get_embeddings([sentence], show_progress=False)[0]

    def embed_jsonl(
        self,
        input_path: str | Path,
        output_path: str | Path,
        *,
        text_key: str = "doc",
        id_key: str = "sample_id",
    ) -> None:
        records = _read_jsonl(input_path, text_key=text_key, id_key=id_key)
        embeddings = self.get_embeddings([record[text_key] for record in records])
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output,
            sample_ids=np.asarray([record[id_key] for record in records], dtype=np.str_),
            embeddings=embeddings,
        )


class VLLMEmbeddingOfflineExtractor:
    """Embedding extractor using vLLM's Python offline API."""

    def __init__(
        self,
        model: str,
        *,
        max_len: int = 1024,
        batch_size: int = 64,
        dtype: str = "float16",
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        pooling_type: str = "MEAN",
        trust_remote_code: bool = False,
    ):
        _validate_model_args(model, max_len, batch_size, tensor_parallel_size, gpu_memory_utilization)
        try:
            from vllm import LLM
            from vllm.config import PoolerConfig
        except ImportError as exc:
            raise ImportError(
                "VLLMEmbeddingOfflineExtractor requires the vllm package"
            ) from exc

        self.batch_size = batch_size
        self.normalize = True
        self.llm = LLM(
            model=model,
            runner="pooling",
            convert="embed",
            max_model_len=max_len,
            dtype=dtype,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            pooler_config=PoolerConfig(pooling_type=pooling_type),
            trust_remote_code=trust_remote_code,
        )

    def get_embeddings(
        self,
        sentences: Sequence[str],
        *,
        batch_size: int | None = None,
        show_progress: bool = True,
    ) -> np.ndarray:
        texts = list(sentences)
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        size = batch_size or self.batch_size
        if size <= 0:
            raise ValueError("batch_size must be positive")

        vectors = []
        with tqdm(
            total=len(texts),
            desc="vLLM offline embeddings",
            unit="row",
            disable=not show_progress,
        ) as progress:
            for start in range(0, len(texts), size):
                outputs = self.llm.embed(texts[start : start + size])
                vectors.extend(output.outputs.embedding for output in outputs)
                progress.update(len(outputs))
        return _as_embedding_matrix(vectors, len(texts), self.normalize)

    def get_embedding(self, sentence: str) -> np.ndarray:
        return self.get_embeddings([sentence], show_progress=False)[0]

    def embed_jsonl(
        self,
        input_path: str | Path,
        output_path: str | Path,
        *,
        text_key: str = "doc",
        id_key: str = "sample_id",
    ) -> None:
        records = _read_jsonl(input_path, text_key=text_key, id_key=id_key)
        embeddings = self.get_embeddings([record[text_key] for record in records])
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output,
            sample_ids=np.asarray([record[id_key] for record in records], dtype=np.str_),
            embeddings=embeddings,
        )


class VLLMGenerationProbabilityExtractor:
    """Generation-probability extractor using vLLM's Python offline API."""

    def __init__(
        self,
        model: str,
        *,
        max_len: int = 1024,
        batch_size: int = 8,
        dtype: str = "float16",
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        trust_remote_code: bool = False,
    ):
        _validate_model_args(model, max_len, batch_size, tensor_parallel_size, gpu_memory_utilization)
        try:
            from vllm import LLM, SamplingParams
        except ImportError as exc:
            raise ImportError(
                "VLLMGenerationProbabilityExtractor requires the vllm package"
            ) from exc

        self.max_len = max_len
        self.batch_size = batch_size
        self.llm = LLM(
            model=model,
            runner="generate",
            max_model_len=max_len,
            dtype=dtype,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=trust_remote_code,
        )
        self.sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=1,
            prompt_logprobs=1,
        )

    @staticmethod
    def _mean_prompt_logprob_from_parts(token_ids, prompt_logprobs) -> float:
        if not token_ids or not prompt_logprobs:
            raise RuntimeError("vLLM did not return prompt log probabilities")
        if len(token_ids) != len(prompt_logprobs):
            raise RuntimeError(
                "vLLM prompt logprob length does not match prompt token length: "
                f"tokens={len(token_ids)}, logprobs={len(prompt_logprobs)}"
            )

        values = []
        for token_id, position_logprobs in zip(token_ids[1:], prompt_logprobs[1:]):
            if position_logprobs is None:
                raise RuntimeError("vLLM returned a missing prompt logprob")
            token_logprob = position_logprobs.get(token_id)
            if token_logprob is None:
                token_logprob = position_logprobs.get(str(token_id))
            if token_logprob is None:
                raise RuntimeError(
                    f"vLLM did not return the selected prompt token {token_id}"
                )
            value = getattr(token_logprob, "logprob", token_logprob)
            if isinstance(value, dict):
                value = value["logprob"]
            value = float(value)
            if not np.isfinite(value):
                raise RuntimeError("vLLM returned a non-finite prompt logprob")
            values.append(value)

        if not values:
            raise RuntimeError("No prompt token logprob was available")
        return float(np.mean(values))

    @classmethod
    def _mean_prompt_logprob(cls, request_output) -> float:
        return cls._mean_prompt_logprob_from_parts(
            request_output.prompt_token_ids,
            request_output.prompt_logprobs,
        )

    def get_log_probabilities(
        self,
        sentences: Sequence[str],
        *,
        batch_size: int | None = None,
        show_progress: bool = True,
    ) -> np.ndarray:
        texts = list(sentences)
        if not texts:
            return np.empty((0,), dtype=np.float32)
        size = batch_size or self.batch_size
        if size <= 0:
            raise ValueError("batch_size must be positive")

        scores = []
        with tqdm(
            total=len(texts),
            desc="vLLM offline generation probabilities",
            unit="row",
            disable=not show_progress,
        ) as progress:
            for start in range(0, len(texts), size):
                batch = texts[start : start + size]
                outputs = self.llm.generate(
                    batch,
                    self.sampling_params,
                    use_tqdm=False,
                )
                if len(outputs) != len(batch):
                    raise RuntimeError(
                        "vLLM output length does not match input batch: "
                        f"inputs={len(batch)}, outputs={len(outputs)}"
                    )
                scores.extend(self._mean_prompt_logprob(output) for output in outputs)
                progress.update(len(batch))
        return np.asarray(scores, dtype=np.float32)

    def cal_gen_prob(self, sentence: str) -> float:
        return float(self.get_log_probabilities([sentence], show_progress=False)[0])

    def score_jsonl(
        self,
        input_path: str | Path,
        output_path: str | Path,
        *,
        text_key: str = "doc",
        id_key: str = "sample_id",
    ) -> None:
        _score_jsonl(self, input_path, output_path, text_key=text_key, id_key=id_key)


class VLLMGenerationProbabilityAPIExtractor:
    """Generation-probability client for vLLM's completions API."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8000/v1",
        model: str | None = None,
        api_key: str | None = None,
        max_len: int = 1024,
        batch_size: int = 8,
        timeout: float = 600.0,
        max_retries: int = 5,
        max_workers: int = 4,
    ):
        if not model:
            raise ValueError("vLLM generation model name is required")
        if max_len <= 1:
            raise ValueError("max_len must be greater than 1")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        if max_retries < 0:
            raise ValueError("max_retries cannot be negative")
        if max_workers <= 0:
            raise ValueError("max_workers must be positive")

        self.endpoint = _endpoint(base_url, "/completions")
        self.model = model
        self.api_key = api_key
        self.max_len = max_len
        self.batch_size = batch_size
        self.timeout = timeout
        self.max_retries = max_retries
        self.max_workers = max_workers

    def _post_batch(self, sentences: Sequence[str]) -> np.ndarray:
        payload = {
            "model": self.model,
            "prompt": list(sentences),
            "max_tokens": 1,
            "temperature": 0.0,
            "prompt_logprobs": 1,
            "return_token_ids": True,
            "truncate_prompt_tokens": self.max_len - 1,
            "stream": False,
        }
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = Request(
            self.endpoint,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        response = _post_json(
            request,
            label="completion",
            timeout=self.timeout,
            max_retries=self.max_retries,
        )
        choices = response.get("choices")
        if not isinstance(choices, list) or len(choices) != len(sentences):
            raise RuntimeError(
                "vLLM completion response length does not match request: "
                f"requested={len(sentences)}, "
                f"returned={len(choices) if isinstance(choices, list) else 'invalid'}"
            )

        ordered = sorted(
            enumerate(choices),
            key=lambda item: int(item[1].get("index", item[0])),
        )
        scores = []
        for _, choice in ordered:
            token_ids = choice.get("prompt_token_ids")
            prompt_logprobs = choice.get("prompt_logprobs")
            if token_ids is None or prompt_logprobs is None:
                raise RuntimeError(
                    "vLLM did not return prompt_token_ids/prompt_logprobs; "
                    "check the server version and request support"
                )
            scores.append(
                VLLMGenerationProbabilityExtractor._mean_prompt_logprob_from_parts(
                    token_ids,
                    prompt_logprobs,
                )
            )
        return np.asarray(scores, dtype=np.float32)

    def get_log_probabilities(
        self,
        sentences: Sequence[str],
        *,
        batch_size: int | None = None,
        show_progress: bool = True,
    ) -> np.ndarray:
        texts = list(sentences)
        if not texts:
            return np.empty((0,), dtype=np.float32)
        size = batch_size or self.batch_size
        if size <= 0:
            raise ValueError("batch_size must be positive")

        batches = [texts[start : start + size] for start in range(0, len(texts), size)]
        scores = []
        with tqdm(
            total=len(texts),
            desc="vLLM API generation probabilities",
            unit="row",
            disable=not show_progress,
        ) as progress:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                for batch, batch_scores in zip(
                    batches,
                    executor.map(self._post_batch, batches),
                ):
                    scores.append(batch_scores)
                    progress.update(len(batch))
        return np.concatenate(scores, axis=0)

    def score_jsonl(
        self,
        input_path: str | Path,
        output_path: str | Path,
        *,
        text_key: str = "doc",
        id_key: str = "sample_id",
    ) -> None:
        _score_jsonl(self, input_path, output_path, text_key=text_key, id_key=id_key)


class Extractor:
    """Compatibility facade using vLLM's Python interface for both features."""

    def __init__(
        self,
        generator: str = "meta-llama/Llama-3.1-8B",
        embedder: str = "meta-llama/Llama-3.1-8B",
        *,
        max_len: int = 1024,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        dtype: str = "float16",
        trust_remote_code: bool = False,
    ):
        common = {
            "max_len": max_len,
            "tensor_parallel_size": tensor_parallel_size,
            "gpu_memory_utilization": gpu_memory_utilization,
            "dtype": dtype,
            "trust_remote_code": trust_remote_code,
        }
        self._embedder = VLLMEmbeddingOfflineExtractor(model=embedder, **common)
        self._generator = VLLMGenerationProbabilityExtractor(model=generator, **common)

    def get_embedding(self, sentence: str, max_len: int | None = None) -> np.ndarray:
        return self._embedder.get_embedding(sentence)

    def cal_gen_prob(self, sentence: str, max_len: int | None = None) -> float:
        return self._generator.cal_gen_prob(sentence)


def _validate_model_args(
    model: str,
    max_len: int,
    batch_size: int,
    tensor_parallel_size: int,
    gpu_memory_utilization: float,
) -> None:
    if not model:
        raise ValueError("vLLM model name is required")
    if max_len <= 1:
        raise ValueError("max_len must be greater than 1")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if tensor_parallel_size <= 0:
        raise ValueError("tensor_parallel_size must be positive")
    if not 0 < gpu_memory_utilization <= 1:
        raise ValueError("gpu_memory_utilization must be in (0, 1]")


def _read_jsonl(
    input_path: str | Path,
    *,
    text_key: str,
    id_key: str,
) -> list[dict[str, Any]]:
    records = []
    with Path(input_path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"{input_path}:{line_number} is not an object")
            if not isinstance(record.get(id_key), str):
                raise ValueError(f"{input_path}:{line_number} has no string {id_key!r}")
            if not isinstance(record.get(text_key), str):
                raise ValueError(f"{input_path}:{line_number} has no string {text_key!r}")
            records.append(record)
    return records


def _score_jsonl(
    extractor,
    input_path: str | Path,
    output_path: str | Path,
    *,
    text_key: str,
    id_key: str,
) -> None:
    sample_ids = []
    scores = []
    batch_ids = []
    batch_texts = []

    def flush_batch() -> None:
        if not batch_texts:
            return
        batch_scores = extractor.get_log_probabilities(
            batch_texts,
            show_progress=False,
        )
        sample_ids.extend(batch_ids)
        scores.extend(batch_scores.tolist())
        batch_ids.clear()
        batch_texts.clear()

    with tqdm(desc="Read and score generation probabilities", unit="row") as progress:
        with Path(input_path).open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError(f"{input_path}:{line_number} is not an object")
                if not isinstance(record.get(id_key), str):
                    raise ValueError(f"{input_path}:{line_number} has no string {id_key!r}")
                if not isinstance(record.get(text_key), str):
                    raise ValueError(f"{input_path}:{line_number} has no string {text_key!r}")
                batch_ids.append(record[id_key])
                batch_texts.append(record[text_key])
                progress.update(1)
                if len(batch_texts) >= extractor.batch_size:
                    flush_batch()
        flush_batch()

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        sample_ids=np.asarray(sample_ids, dtype=np.str_),
        ln_probability=np.asarray(scores, dtype=np.float32),
    )


def _main_vllm() -> None:
    parser = argparse.ArgumentParser(description="Run batch extraction through vLLM.")
    parser.add_argument(
        "--mode",
        choices=("embedding", "embedding-offline", "generation-probability", "generation-probability-api"),
        default="embedding",
        help="embedding uses HTTP API; embedding-offline uses vLLM's Python API",
    )
    parser.add_argument("--input-jsonl", required=True, type=Path)
    parser.add_argument("--output-npz", required=True, type=Path)
    parser.add_argument(
        "--base-url",
        default=os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8000/v1"),
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("VLLM_MODEL")
        or os.environ.get("VLLM_EMBEDDING_MODEL"),
    )
    parser.add_argument("--api-key", default=os.environ.get("VLLM_API_KEY"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--no-normalize", action="store_true")
    parser.add_argument("--max-len", type=int, default=1024)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--pooling-type", default="MEAN")
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()

    if args.mode == "embedding":
        extractor = VLLMEmbeddingExtractor(
            base_url=args.base_url,
            model=args.model,
            api_key=args.api_key,
            batch_size=args.batch_size,
            timeout=args.timeout,
            max_retries=args.max_retries,
            normalize=not args.no_normalize,
            max_len=args.max_len,
            max_workers=args.num_workers,
        )
        extractor.embed_jsonl(args.input_jsonl, args.output_npz)
    elif args.mode == "embedding-offline":
        extractor = VLLMEmbeddingOfflineExtractor(
            model=args.model,
            max_len=args.max_len,
            batch_size=args.batch_size,
            dtype=args.dtype,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            pooling_type=args.pooling_type,
            trust_remote_code=args.trust_remote_code,
        )
        extractor.embed_jsonl(args.input_jsonl, args.output_npz)
    elif args.mode == "generation-probability":
        extractor = VLLMGenerationProbabilityExtractor(
            model=args.model,
            max_len=args.max_len,
            batch_size=args.batch_size,
            dtype=args.dtype,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            trust_remote_code=args.trust_remote_code,
        )
        extractor.score_jsonl(args.input_jsonl, args.output_npz)
    else:
        extractor = VLLMGenerationProbabilityAPIExtractor(
            base_url=args.base_url,
            model=args.model,
            api_key=args.api_key,
            max_len=args.max_len,
            batch_size=args.batch_size,
            timeout=args.timeout,
            max_retries=args.max_retries,
            max_workers=args.num_workers,
        )
        extractor.score_jsonl(args.input_jsonl, args.output_npz)
    print(f"Saved {args.mode} results to {args.output_npz}")


if __name__ == "__main__":
    _main_vllm()
