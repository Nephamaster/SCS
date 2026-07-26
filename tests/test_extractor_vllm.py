import json
import sys
import types
import unittest
from unittest.mock import patch

import numpy as np

from src.extractor import (
    VLLMEmbeddingExtractor,
    VLLMEmbeddingOfflineExtractor,
    VLLMGenerationProbabilityExtractor,
    VLLMGenerationProbabilityAPIExtractor,
)


class _Response:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


class _Logprob:
    def __init__(self, value):
        self.logprob = value


class _GenerationOutput:
    prompt_token_ids = [10, 20, 30]
    prompt_logprobs = [
        None,
        {20: _Logprob(-1.0)},
        {30: _Logprob(-3.0)},
    ]


class _PoolingOutput:
    def __init__(self, embedding):
        self.outputs = types.SimpleNamespace(embedding=embedding)


class VLLMEmbeddingExtractorTest(unittest.TestCase):
    def test_batches_requests_and_restores_response_order(self):
        responses = [
            _Response(
                {
                    "data": [
                        {"index": 1, "embedding": [0.0, 2.0]},
                        {"index": 0, "embedding": [3.0, 0.0]},
                    ]
                }
            ),
            _Response({"data": [{"index": 0, "embedding": [1.0, 1.0]}]}),
        ]
        with patch("src.extractor.urlopen", side_effect=responses) as urlopen:
            extractor = VLLMEmbeddingExtractor(
                base_url="http://localhost:8000/v1",
                model="test-embedder",
                batch_size=2,
                max_len=4096,
                max_workers=1,
            )
            embeddings = extractor.get_embeddings(
                ["first", "second", "third"],
                show_progress=False,
            )

        self.assertEqual(urlopen.call_count, 2)
        first_payload = json.loads(urlopen.call_args_list[0].args[0].data)
        self.assertEqual(first_payload["model"], "test-embedder")
        self.assertEqual(first_payload["input"], ["first", "second"])
        self.assertEqual(first_payload["truncate_prompt_tokens"], 4096)
        self.assertEqual(first_payload["truncation_side"], "right")
        self.assertEqual(embeddings.shape, (3, 2))
        self.assertTrue(np.allclose(np.linalg.norm(embeddings, axis=1), 1.0))
        self.assertTrue(np.allclose(embeddings[0], [1.0, 0.0]))
        self.assertTrue(np.allclose(embeddings[1], [0.0, 1.0]))

    def test_offline_embedding_uses_vllm_python_api(self):
        class FakeLLM:
            init_kwargs = None

            def __init__(self, **kwargs):
                FakeLLM.init_kwargs = kwargs

            def embed(self, texts):
                return [_PoolingOutput([3.0, 4.0]) for _ in texts]

        fake_vllm = types.ModuleType("vllm")
        fake_vllm.LLM = FakeLLM
        fake_config = types.ModuleType("vllm.config")
        fake_config.PoolerConfig = lambda **kwargs: kwargs

        with patch.dict(
            sys.modules,
            {"vllm": fake_vllm, "vllm.config": fake_config},
        ):
            extractor = VLLMEmbeddingOfflineExtractor(
                model="test-embedder",
                batch_size=2,
            )
            embeddings = extractor.get_embeddings(
                ["first", "second", "third"],
                show_progress=False,
            )

        self.assertEqual(FakeLLM.init_kwargs["runner"], "pooling")
        self.assertEqual(FakeLLM.init_kwargs["convert"], "embed")
        self.assertEqual(FakeLLM.init_kwargs["pooler_config"], {"pooling_type": "MEAN"})
        self.assertEqual(embeddings.shape, (3, 2))
        self.assertTrue(np.allclose(embeddings, [[0.6, 0.8]] * 3))

    def test_generation_probability_matches_shifted_token_mean(self):
        score = VLLMGenerationProbabilityExtractor._mean_prompt_logprob(
            _GenerationOutput()
        )
        self.assertAlmostEqual(score, -2.0)

    def test_generation_probability_api_reads_prompt_logprobs(self):
        responses = [
            _Response(
                {
                    "choices": [
                        {
                            "index": 0,
                            "prompt_token_ids": [10, 20, 30],
                            "prompt_logprobs": [
                                None,
                                {"20": {"logprob": -1.0}},
                                {"30": {"logprob": -3.0}},
                            ],
                        }
                    ]
                }
            )
        ]
        with patch("src.extractor.urlopen", side_effect=responses) as urlopen:
            extractor = VLLMGenerationProbabilityAPIExtractor(
                base_url="http://localhost:8001/v1",
                model="test-generator",
                batch_size=2,
                max_len=4096,
            )
            scores = extractor.get_log_probabilities(
                ["hello"],
                show_progress=False,
            )

        self.assertEqual(urlopen.call_count, 1)
        payload = json.loads(urlopen.call_args.args[0].data)
        self.assertEqual(payload["model"], "test-generator")
        self.assertEqual(payload["prompt"], ["hello"])
        self.assertEqual(payload["prompt_logprobs"], 1)
        self.assertEqual(payload["truncate_prompt_tokens"], 4095)
        self.assertTrue(np.allclose(scores, [-2.0]))


if __name__ == "__main__":
    unittest.main()
