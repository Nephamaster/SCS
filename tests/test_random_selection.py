import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from data_selection.random_select import (
    build_indices,
    build_sft_record,
    load_candidate_records,
    main,
)


class RandomSelectionTest(unittest.TestCase):
    def test_candidate_pool_records_keep_indexable_identity(self):
        record = {
            "sample_id": "tulu::example::0",
            "source": "tulu",
            "messages": [
                {"role": "user", "content": "Question"},
                {"role": "assistant", "content": "Answer"},
            ],
        }
        self.assertEqual(
            build_sft_record(record, 17),
            {
                "candidate_index": 17,
                "sample_id": "tulu::example::0",
                "source": "tulu",
                "messages": record["messages"],
            },
        )

    def test_loader_reads_candidate_messages_jsonl(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "candidate_messages.jsonl"
            records = [
                {
                    "sample_id": f"sample-{index}",
                    "source": "fixture",
                    "messages": [
                        {"role": "user", "content": f"Question {index}"},
                        {"role": "assistant", "content": f"Answer {index}"},
                    ],
                }
                for index in range(2)
            ]
            input_path.write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )

            loaded = load_candidate_records(input_path)

        self.assertEqual(loaded, records)

    def test_random_groups_are_reproducible_and_bounded(self):
        groups, seeds = build_indices(
            source_size=100,
            num_groups=3,
            sample_size=10,
            base_seed=123,
            seed_step=1,
            disjoint=False,
        )
        repeated_groups, repeated_seeds = build_indices(
            source_size=100,
            num_groups=3,
            sample_size=10,
            base_seed=123,
            seed_step=1,
            disjoint=False,
        )

        self.assertEqual(groups, repeated_groups)
        self.assertEqual(seeds, repeated_seeds)
        for group in groups:
            self.assertEqual(len(group), 10)
            self.assertEqual(len(set(group)), 10)
            self.assertTrue(all(0 <= index < 100 for index in group))

    def test_main_writes_indexed_sft_only(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_path = root / "candidate_messages.jsonl"
            output_dir = root / "sft" / "random"
            records = [
                {
                    "sample_id": f"sample-{index}",
                    "source": "fixture",
                    "messages": [
                        {"role": "user", "content": f"Question {index}"},
                        {"role": "assistant", "content": f"Answer {index}"},
                    ],
                }
                for index in range(4)
            ]
            input_path.write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )

            argv = [
                "random_select.py",
                "--input",
                str(input_path),
                "--sft_output_dir",
                str(output_dir),
                "--num_groups",
                "1",
                "--sample_size",
                "2",
                "--seed",
                "123",
            ]
            with patch.object(sys, "argv", argv):
                main()

            output_path = output_dir / "random_01.jsonl"
            output_records = [
                json.loads(line)
                for line in output_path.read_text(encoding="utf-8").splitlines()
            ]
            manifest = json.loads(
                (output_dir / "random_manifest.json").read_text(encoding="utf-8")
            )

        self.assertEqual(len(output_records), 2)
        self.assertEqual(
            [record["candidate_index"] for record in output_records],
            manifest["groups"][0]["candidate_indices"],
        )
        self.assertTrue(all("messages" in record for record in output_records))
        self.assertFalse((output_dir / "random_01.json").exists())


if __name__ == "__main__":
    unittest.main()
