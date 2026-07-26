import json
import unittest
from unittest.mock import patch

import numpy as np

from src.extractor import (
    VLLMEmbeddingExtractor,
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
            )
            embeddings = extractor.get_embeddings(
                ["first", "second", "third"],
                show_progress=False,
            )

        self.assertEqual(urlopen.call_count, 2)
        first_payload = json.loads(urlopen.call_args_list[0].args[0].data)
        self.assertEqual(first_payload["model"], "test-embedder")
        self.assertEqual(first_payload["input"], ["first", "second"])
        self.assertEqual(embeddings.shape, (3, 2))
        self.assertTrue(np.allclose(np.linalg.norm(embeddings, axis=1), 1.0))
        self.assertTrue(np.allclose(embeddings[0], [1.0, 0.0]))
        self.assertTrue(np.allclose(embeddings[1], [0.0, 1.0]))

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
        self.assertTrue(np.allclose(scores, [-2.0]))


if __name__ == "__main__":
    unittest.main()
