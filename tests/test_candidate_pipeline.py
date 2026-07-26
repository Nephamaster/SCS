import tempfile
import unittest
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from src.candidate.audit_candidate import audit
from src.candidate.build_candidate import apply_dedup, build
from src.candidate.quota import calculate_source_quotas
from src.data.dedup import character_ngrams, exact_deduplicate, minhash_deduplicate
from src.data.normalize import validate_messages
from src.data.oasst2_builder import build_oasst2_validation


class CandidatePipelineTest(unittest.TestCase):
    def test_quota_sum_and_bounds(self):
        counts = {"source_a": 1000, "source_b": 1000, "coig_cqia": 100}
        quotas = calculate_source_quotas(counts, 1500)
        self.assertEqual(sum(quotas.values()), 1500)
        for source, quota in quotas.items():
            self.assertGreater(quota, 0)
            self.assertLessEqual(quota, counts[source])

    def test_priority_and_near_duplicate_deduplication(self):
        records = [
            {
                "sample_id": "tulu",
                "source": "tulu",
                "metadata": {},
                "_dedup_text": "same",
            },
            {
                "sample_id": "coig",
                "source": "coig_cqia",
                "metadata": {"human_verified": True},
                "_dedup_text": "same",
            },
        ]
        priority = lambda record: (
            0 if record["sample_id"] == "coig" else 2,
            record["source"],
            record["sample_id"],
        )
        kept, removed = exact_deduplicate(
            records, lambda record: record["_dedup_text"], priority
        )
        self.assertEqual([record["sample_id"] for record in kept], ["coig"])
        self.assertEqual(removed[0]["matched_sample_id"], "coig")

        near_records = [
            {"sample_id": "a", "source": "a", "metadata": {}, "_dedup_text": "abcdefghij"},
            {"sample_id": "b", "source": "b", "metadata": {}, "_dedup_text": "abcdefghiX"},
        ]
        kept, _ = minhash_deduplicate(
            near_records,
            lambda record: record["_dedup_text"],
            lambda record: (0, record["source"], record["sample_id"]),
            lambda text: character_ngrams(text, 5),
            threshold=0.7,
        )
        self.assertEqual(len(kept), 1)

    def test_skip_minhash_dedup_does_not_require_tokenizer(self):
        records = [
            {
                "sample_id": "a",
                "source": "tulu",
                "metadata": {},
                "_dedup_text": "abcdefghij",
            },
            {
                "sample_id": "b",
                "source": "tulu",
                "metadata": {},
                "_dedup_text": "abcdefghiX",
            },
        ]
        args = SimpleNamespace(skip_minhash_dedup=True)
        kept, removed = apply_dedup(records, args, Counter())
        self.assertEqual(len(kept), 2)
        self.assertEqual(removed, [])

    def test_complete_candidate_pool_is_reused_without_overwrite(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            candidate_dir = Path(temp_dir) / "candidate"
            removals_dir = candidate_dir / "removals"
            removals_dir.mkdir(parents=True)
            (candidate_dir / "candidate_manifest.json").write_text(
                '{"sample_count": 10}\n', encoding="utf-8"
            )
            for name in (
                "candidate.canonical.jsonl",
                "candidate_messages.jsonl",
                "candidate_sft.jsonl",
                "candidate_doc.jsonl",
                "candidate_metadata.jsonl",
            ):
                (candidate_dir / name).touch()
            for name in (
                "oasst2_decontamination.jsonl",
                "training_deduplication.jsonl",
            ):
                (removals_dir / name).touch()

            args = SimpleNamespace(candidate_dir=candidate_dir, overwrite=False)
            with patch("src.candidate.build_candidate.load_and_adapt") as load_and_adapt:
                manifest = build(args)

            load_and_adapt.assert_not_called()
            self.assertEqual(manifest["sample_count"], 10)

    def test_oasst_path_and_tool_message_validation(self):
        rows = [
            {
                "message_id": "root",
                "parent_id": None,
                "message_tree_id": "tree",
                "text": "question",
                "role": "prompter",
                "rank": None,
                "lang": "en",
                "review_result": True,
                "deleted": False,
                "tree_state": "ready_for_export",
            },
            {
                "message_id": "answer",
                "parent_id": "root",
                "message_tree_id": "tree",
                "text": "answer",
                "role": "assistant",
                "rank": 0,
                "lang": "en",
                "review_result": True,
                "deleted": False,
                "tree_state": "ready_for_export",
            },
        ]
        records, removed = build_oasst2_validation(rows)
        self.assertEqual(len(records), 1)
        self.assertFalse(removed)
        valid, reason = validate_messages(
            [
                {"role": "user", "content": "question"},
                {"role": "tool", "content": "{}"},
                {"role": "assistant", "content": "answer"},
            ]
        )
        self.assertTrue(valid, reason)

    def test_end_to_end_candidate_exports_and_audit(self):
        tulu_rows = [
            {
                "id": f"tulu-{index}",
                "source": "tulu-fixture",
                "messages": [
                    {"role": "user", "content": f"Question {index}"},
                    {"role": "assistant", "content": f"Answer {index}"},
                ],
            }
            for index in range(8)
        ]
        coig_rows = [
            {
                "id": f"coig-{index}",
                "instruction": f"Instruction {index}",
                "input": None,
                "output": f"Output {index}",
                "human_verified": index == 0,
            }
            for index in range(2)
        ]
        oasst_rows = [
            {
                "message_id": "root",
                "parent_id": None,
                "message_tree_id": "fixture-tree",
                "text": "A validation question",
                "role": "prompter",
                "rank": None,
                "lang": "en",
                "review_result": True,
                "deleted": False,
                "tree_state": "ready_for_export",
            },
            {
                "message_id": "answer",
                "parent_id": "root",
                "message_tree_id": "fixture-tree",
                "text": "A validation answer",
                "role": "assistant",
                "rank": 0,
                "lang": "en",
                "review_result": True,
                "deleted": False,
                "tree_state": "ready_for_export",
            },
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            paths = {
                "tulu": root / "tulu",
                "coig": root / "coig",
                "oasst": root / "oasst",
            }
            candidate_dir = root / "candidate"
            args = SimpleNamespace(
                tulu_path=paths["tulu"],
                coig_path=paths["coig"],
                oasst_path=paths["oasst"],
                candidate_dir=candidate_dir,
                dev_dir=root / "dev",
                normalized_dir=root / "normalized",
                target_count=10,
                seed=42,
                candidate_version="test",
                minhash_tokenizer_model=None,
                doc_minhash_tokenization="character",
                minhash_window_size=5,
                minhash_permutations=256,
                minhash_threshold=0.80,
                prompt_threshold=0.80,
                skip_minhash_dedup=False,
                skip_oasst2_decontamination=False,
                overwrite=False,
            )
            with patch(
                "src.candidate.build_candidate.read_records",
                side_effect=[tulu_rows, coig_rows, oasst_rows],
            ) as read_records:
                manifest = build(args)
                self.assertEqual(read_records.call_count, 3)
                self.assertEqual(
                    read_records.call_args_list[2].kwargs["split_name"],
                    "validation",
                )
            self.assertEqual(manifest["sample_count"], 10)
            self.assertEqual(manifest["stage_counts"]["candidate"], 10)
            self.assertTrue((root / "normalized" / "train_merged.jsonl").exists())
            result = audit(candidate_dir)
            self.assertTrue(result["aligned_outputs"])
            self.assertEqual(result["sample_count"], 10)


if __name__ == "__main__":
    unittest.main()
