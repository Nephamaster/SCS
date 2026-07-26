import argparse
import json
import os
import time
from pathlib import Path
from typing import Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
from tqdm.auto import tqdm


class Extractor:
    def __init__(self, generator:str="meta-llama/Llama-3.1-8B", embedder:str="meta-llama/Llama-3.1-8B"):
        import torch
        import torch.nn.functional as F
        from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel

        self._torch = torch
        self._F = F
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(generator, trust_remote_code=True)
        if 'llama' in generator.lower():
            tokenizer.pad_token = tokenizer.eos_token
        self.gen_tokenizer = tokenizer
        self.gen_model = AutoModelForCausalLM.from_pretrained(
            generator, device_map="auto", dtype=torch.float16, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(embedder, trust_remote_code=True)
        if 'llama' in embedder.lower():
            tokenizer.pad_token = tokenizer.eos_token
        self.emb_tokenizer = tokenizer
        self.emb_model = AutoModel.from_pretrained(
            embedder, device_map="auto", dtype=torch.float16, trust_remote_code=True)
        self.emb_model.eval()

    def cal_gen_prob(self, sentence:str, max_len:int=4096) -> float:
        """计算输入文本的生成概率"""
        torch = self._torch
        F = self._F
        self.gen_tokenizer.padding_side = "right"
        inputs = self.gen_tokenizer(
            sentence, 
            padding=True, 
            truncation=True,
            max_length = max_len, 
            return_tensors='pt'
        ).to(self.device)
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        labels = input_ids.clone()
        with torch.inference_mode():
            outputs = self.gen_model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
            logits = outputs.logits # [1, seq_len, vocab_size]
        torch.cuda.empty_cache()
        shift_logits = logits[:, :-1, :]  # [1, seq_len-1, vocab_size]
        shift_labels = labels[:, 1:]      # [1, seq_len-1]
        # print('output length: ', outputs.logits.size(1))
        log_softmax = F.log_softmax(shift_logits, dim=-1)
        log_likes = log_softmax.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)  # [1, seq_len-1]
        log_likes_norm = log_likes.mean().item()
        return log_likes_norm
    
    def get_embedding(self, sentence:str, max_len:int=256) -> np.ndarray:
        """获取输入文本的语义向量"""
        torch = self._torch
        encoded_input = self.emb_tokenizer(
            [sentence],
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors='pt',
            return_attention_mask=True
        ).to(self.device)
        
        with torch.inference_mode():
            model_output = self.emb_model(**encoded_input)
            hidden_states = model_output.last_hidden_state  # [1, seq_len, hidden_size]
            attention_mask = encoded_input['attention_mask']
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            sum_embeddings = torch.sum(hidden_states * input_mask_expanded, 1)
            sum_mask = input_mask_expanded.sum(1)
            sum_mask = torch.clamp(sum_mask, min=1e-9)  # 防止除零
            sentence_embeddings = sum_embeddings / sum_mask
            sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
        
        torch.cuda.empty_cache()
        return sentence_embeddings.cpu().numpy()[0]


class VLLMEmbeddingExtractor:
    """Batch client for a vLLM OpenAI-compatible embeddings endpoint.

    This client only requests embeddings. It does not perform token counting,
    truncation, generation-probability calculation, or local model loading.
    """

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8000/v1",
        model: str | None = None,
        api_key: str | None = None,
        batch_size: int = 64,
        timeout: float = 600.0,
        max_retries: int = 5,
        normalize: bool = True,
    ):
        if not model:
            raise ValueError("vLLM embedding model name is required")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        if max_retries < 0:
            raise ValueError("max_retries cannot be negative")

        base_url = base_url.rstrip("/")
        if base_url.endswith("/embeddings"):
            self.endpoint = base_url
        elif base_url.endswith("/v1"):
            self.endpoint = base_url + "/embeddings"
        else:
            self.endpoint = base_url + "/v1/embeddings"
        self.model = model
        self.api_key = api_key
        self.batch_size = batch_size
        self.timeout = timeout
        self.max_retries = max_retries
        self.normalize = normalize

    def _post_batch(self, sentences: Sequence[str]) -> np.ndarray:
        payload = {
            "model": self.model,
            "input": list(sentences),
            "encoding_format": "float",
        }
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        request = Request(
            self.endpoint,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        response = None
        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                with urlopen(request, timeout=self.timeout) as handle:
                    response = json.loads(handle.read().decode("utf-8"))
                break
            except HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace")
                message = f"vLLM embeddings HTTP {exc.code}: {body[:1000]}"
                if exc.code < 500 and exc.code not in {408, 429}:
                    raise RuntimeError(message) from exc
                last_error = RuntimeError(message)
            except (URLError, TimeoutError, OSError) as exc:
                last_error = exc

            if attempt < self.max_retries:
                time.sleep(min(2 ** attempt, 30))

        if response is None:
            raise RuntimeError(
                f"vLLM embeddings request failed after {self.max_retries + 1} attempts"
            ) from last_error
        if "error" in response:
            raise RuntimeError(f"vLLM embeddings error: {response['error']}")

        data = response.get("data")
        if not isinstance(data, list) or len(data) != len(sentences):
            raise RuntimeError(
                "vLLM embeddings response length does not match request: "
                f"requested={len(sentences)}, returned={len(data) if isinstance(data, list) else 'invalid'}"
            )

        indexed = sorted(
            enumerate(data),
            key=lambda item: int(item[1].get("index", item[0])),
        )
        vectors = np.asarray(
            [item[1]["embedding"] for item in indexed],
            dtype=np.float32,
        )
        if vectors.ndim != 2 or vectors.shape[0] != len(sentences):
            raise RuntimeError(f"Invalid embedding matrix shape: {vectors.shape}")
        if not np.isfinite(vectors).all():
            raise RuntimeError("vLLM returned a non-finite embedding")
        if self.normalize:
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            vectors = vectors / np.clip(norms, 1e-12, None)
        return vectors

    def get_embeddings(
        self,
        sentences: Sequence[str],
        *,
        batch_size: int | None = None,
        show_progress: bool = True,
    ) -> np.ndarray:
        """Return embeddings in exactly the same order as ``sentences``."""

        texts = list(sentences)
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        size = batch_size or self.batch_size
        if size <= 0:
            raise ValueError("batch_size must be positive")

        vectors = []
        progress = tqdm(
            total=len(texts),
            desc="vLLM embeddings",
            unit="row",
            disable=not show_progress,
        )
        try:
            for start in range(0, len(texts), size):
                batch = texts[start : start + size]
                vectors.append(self._post_batch(batch))
                progress.update(len(batch))
        finally:
            progress.close()
        return np.concatenate(vectors, axis=0)

    def get_embedding(self, sentence: str) -> np.ndarray:
        """Compatibility wrapper matching the local ``Extractor`` API."""

        return self.get_embeddings([sentence], show_progress=False)[0]

    def embed_jsonl(
        self,
        input_path: str | Path,
        output_path: str | Path,
        *,
        text_key: str = "doc",
        id_key: str = "sample_id",
    ) -> None:
        """Embed a candidate-doc JSONL and save IDs plus vectors as ``.npz``."""

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

        ids = [record[id_key] for record in records]
        texts = [record[text_key] for record in records]
        embeddings = self.get_embeddings(texts)
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output,
            sample_ids=np.asarray(ids, dtype=np.str_),
            embeddings=embeddings,
        )


class VLLMGenerationProbabilityExtractor:
    """Batch generation-probability extractor using vLLM's offline API.

    ``prompt_logprobs`` contains the model log probability assigned to each
    input token.  The first input token has no preceding context, so it is
    excluded from the mean, matching :meth:`Extractor.cal_gen_prob`.
    """

    def __init__(
        self,
        model: str,
        *,
        max_len: int = 4096,
        batch_size: int = 8,
        dtype: str = "float16",
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        trust_remote_code: bool = False,
    ):
        if not model:
            raise ValueError("vLLM generation model name is required")
        if max_len <= 1:
            raise ValueError("max_len must be greater than 1")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if tensor_parallel_size <= 0:
            raise ValueError("tensor_parallel_size must be positive")
        if not 0 < gpu_memory_utilization <= 1:
            raise ValueError("gpu_memory_utilization must be in (0, 1]")

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
            dtype=dtype,
            max_model_len=max_len,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=trust_remote_code,
        )
        self.tokenizer = self.llm.get_tokenizer()
        self.sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=1,
            prompt_logprobs=1,
        )

    def _tokenize(self, sentence: str) -> list[int]:
        encoded = self.tokenizer(
            sentence,
            truncation=True,
            max_length=self.max_len,
            add_special_tokens=True,
        )
        token_ids = encoded["input_ids"]
        if len(token_ids) <= 1:
            raise ValueError(
                "Generation probability requires at least two input tokens"
            )
        return token_ids

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
        """Return mean input-token log probabilities in input order."""

        texts = list(sentences)
        if not texts:
            return np.empty((0,), dtype=np.float32)
        size = batch_size or self.batch_size
        if size <= 0:
            raise ValueError("batch_size must be positive")

        scores = []
        progress = tqdm(
            total=len(texts),
            desc="vLLM generation probabilities",
            unit="row",
            disable=not show_progress,
        )
        try:
            for start in range(0, len(texts), size):
                batch = texts[start : start + size]
                prompts = [
                    {"prompt_token_ids": self._tokenize(sentence)}
                    for sentence in batch
                ]
                outputs = self.llm.generate(
                    prompts,
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
        finally:
            progress.close()
        return np.asarray(scores, dtype=np.float32)

    def cal_gen_prob(self, sentence: str) -> float:
        """Compatibility wrapper matching :meth:`Extractor.cal_gen_prob`."""

        return float(self.get_log_probabilities([sentence], show_progress=False)[0])

    def score_jsonl(
        self,
        input_path: str | Path,
        output_path: str | Path,
        *,
        text_key: str = "doc",
        id_key: str = "sample_id",
    ) -> None:
        """Score a candidate-doc JSONL and save IDs plus log probabilities."""

        sample_ids = []
        scores = []
        batch_ids = []
        batch_texts = []
        progress = tqdm(desc="Read and score generation probabilities", unit="row")

        def flush_batch() -> None:
            if not batch_texts:
                return
            batch_scores = self.get_log_probabilities(
                batch_texts,
                show_progress=False,
            )
            sample_ids.extend(batch_ids)
            scores.extend(batch_scores.tolist())
            batch_ids.clear()
            batch_texts.clear()

        try:
            with Path(input_path).open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    if not isinstance(record, dict):
                        raise ValueError(f"{input_path}:{line_number} is not an object")
                    if not isinstance(record.get(id_key), str):
                        raise ValueError(
                            f"{input_path}:{line_number} has no string {id_key!r}"
                        )
                    if not isinstance(record.get(text_key), str):
                        raise ValueError(
                            f"{input_path}:{line_number} has no string {text_key!r}"
                        )
                    batch_ids.append(record[id_key])
                    batch_texts.append(record[text_key])
                    progress.update(1)
                    if len(batch_texts) >= self.batch_size:
                        flush_batch()
            flush_batch()
        finally:
            progress.close()

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output,
            sample_ids=np.asarray(sample_ids, dtype=np.str_),
            ln_probability=np.asarray(scores, dtype=np.float32),
        )


class VLLMGenerationProbabilityAPIExtractor:
    """Batch generation-probability client for vLLM's completion API."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8000/v1",
        model: str | None = None,
        api_key: str | None = None,
        max_len: int = 4096,
        batch_size: int = 8,
        timeout: float = 600.0,
        max_retries: int = 5,
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

        base_url = base_url.rstrip("/")
        if base_url.endswith("/completions"):
            self.endpoint = base_url
        elif base_url.endswith("/v1"):
            self.endpoint = base_url + "/completions"
        else:
            self.endpoint = base_url + "/v1/completions"
        self.model = model
        self.api_key = api_key
        self.max_len = max_len
        self.batch_size = batch_size
        self.timeout = timeout
        self.max_retries = max_retries

    def _post_batch(self, sentences: Sequence[str]) -> np.ndarray:
        payload = {
            "model": self.model,
            "prompt": list(sentences),
            "max_tokens": 1,
            "temperature": 0.0,
            "prompt_logprobs": 1,
            "return_token_ids": True,
            "truncate_prompt_tokens": self.max_len,
            "truncation_side": "right",
            "add_special_tokens": True,
            "stream": False,
        }
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        request = Request(
            self.endpoint,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        response = None
        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                with urlopen(request, timeout=self.timeout) as handle:
                    response = json.loads(handle.read().decode("utf-8"))
                break
            except HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace")
                message = f"vLLM completion HTTP {exc.code}: {body[:1000]}"
                if exc.code < 500 and exc.code not in {408, 429}:
                    raise RuntimeError(message) from exc
                last_error = RuntimeError(message)
            except (URLError, TimeoutError, OSError) as exc:
                last_error = exc

            if attempt < self.max_retries:
                time.sleep(min(2 ** attempt, 30))

        if response is None:
            raise RuntimeError(
                f"vLLM completion request failed after {self.max_retries + 1} attempts"
            ) from last_error
        if "error" in response:
            raise RuntimeError(f"vLLM completion error: {response['error']}")

        choices = response.get("choices")
        if not isinstance(choices, list) or len(choices) != len(sentences):
            raise RuntimeError(
                "vLLM completion response length does not match request: "
                f"requested={len(sentences)}, returned={len(choices) if isinstance(choices, list) else 'invalid'}"
            )

        indexed = sorted(
            enumerate(choices),
            key=lambda item: int(item[1].get("index", item[0])),
        )
        scores = []
        for _, choice in indexed:
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

        scores = []
        progress = tqdm(
            total=len(texts),
            desc="vLLM API generation probabilities",
            unit="row",
            disable=not show_progress,
        )
        try:
            for start in range(0, len(texts), size):
                batch = texts[start : start + size]
                scores.append(self._post_batch(batch))
                progress.update(len(batch))
        finally:
            progress.close()
        return np.concatenate(scores, axis=0)

    def score_jsonl(
        self,
        input_path: str | Path,
        output_path: str | Path,
        *,
        text_key: str = "doc",
        id_key: str = "sample_id",
    ) -> None:
        sample_ids = []
        scores = []
        batch_ids = []
        batch_texts = []
        progress = tqdm(desc="Read and score generation probabilities", unit="row")

        def flush_batch() -> None:
            if not batch_texts:
                return
            batch_scores = self.get_log_probabilities(
                batch_texts,
                show_progress=False,
            )
            sample_ids.extend(batch_ids)
            scores.extend(batch_scores.tolist())
            batch_ids.clear()
            batch_texts.clear()

        try:
            with Path(input_path).open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    if not isinstance(record, dict):
                        raise ValueError(f"{input_path}:{line_number} is not an object")
                    if not isinstance(record.get(id_key), str):
                        raise ValueError(
                            f"{input_path}:{line_number} has no string {id_key!r}"
                        )
                    if not isinstance(record.get(text_key), str):
                        raise ValueError(
                            f"{input_path}:{line_number} has no string {text_key!r}"
                        )
                    batch_ids.append(record[id_key])
                    batch_texts.append(record[text_key])
                    progress.update(1)
                    if len(batch_texts) >= self.batch_size:
                        flush_batch()
            flush_batch()
        finally:
            progress.close()

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
        choices=(
            "embedding",
            "generation-probability",
            "generation-probability-api",
        ),
        default="embedding",
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
    parser.add_argument("--max-len", type=int, default=4096)
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()

    if args.mode == "embedding":
        client = VLLMEmbeddingExtractor(
            base_url=args.base_url,
            model=args.model,
            api_key=args.api_key,
            batch_size=args.batch_size,
            timeout=args.timeout,
            max_retries=args.max_retries,
            normalize=not args.no_normalize,
        )
        client.embed_jsonl(args.input_jsonl, args.output_npz)
    elif args.mode == "generation-probability":
        client = VLLMGenerationProbabilityExtractor(
            model=args.model,
            max_len=args.max_len,
            batch_size=args.batch_size,
            dtype=args.dtype,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            trust_remote_code=args.trust_remote_code,
        )
        client.score_jsonl(args.input_jsonl, args.output_npz)
    else:
        client = VLLMGenerationProbabilityAPIExtractor(
            base_url=args.base_url,
            model=args.model,
            api_key=args.api_key,
            max_len=args.max_len,
            batch_size=args.batch_size,
            timeout=args.timeout,
            max_retries=args.max_retries,
        )
        client.score_jsonl(args.input_jsonl, args.output_npz)
    print(f"Saved {args.mode} results to {args.output_npz}")


if __name__ == "__main__":
    _main_vllm()
