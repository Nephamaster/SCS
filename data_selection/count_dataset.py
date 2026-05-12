import argparse
import json
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]


def resolve_dataset_path(dataset_or_path: str) -> Path:
    path = Path(dataset_or_path)
    if path.exists():
        return path

    candidates = [
        ROOT_DIR / "data" / "raw" / f"{dataset_or_path}.json",
        ROOT_DIR / "data" / f"{dataset_or_path}.json",
        ROOT_DIR / "data" / "raw" / f"{dataset_or_path}.jsonl",
        ROOT_DIR / "data" / f"{dataset_or_path}.jsonl",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "Dataset not found. Pass a file path, or a dataset name under data/raw or data."
    )


def count_jsonl(path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def count_json_array(path: Path) -> int:
    count = 0
    depth = 0
    in_string = False
    escape = False
    seen_array = False
    seen_item_at_depth_one = False

    with path.open("r", encoding="utf-8") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break

            for char in chunk:
                if in_string:
                    if escape:
                        escape = False
                    elif char == "\\":
                        escape = True
                    elif char == '"':
                        in_string = False
                    continue

                if char == '"':
                    in_string = True
                    if depth == 1:
                        seen_item_at_depth_one = True
                    continue

                if char.isspace():
                    continue

                if char == "[":
                    depth += 1
                    if depth == 1:
                        seen_array = True
                    elif depth == 2:
                        seen_item_at_depth_one = True
                    continue

                if char == "{":
                    depth += 1
                    if depth == 2:
                        seen_item_at_depth_one = True
                    continue

                if char in "]}":
                    if depth == 1 and char == "]":
                        return count + int(seen_item_at_depth_one)
                    depth -= 1
                    continue

                if char == "," and depth == 1:
                    count += 1
                    seen_item_at_depth_one = False
                    continue

                if depth == 1:
                    seen_item_at_depth_one = True

    if not seen_array:
        raise ValueError(f"{path} is not a JSON array.")
    return count + int(seen_item_at_depth_one)


def count_json(path: Path) -> int:
    with path.open("r", encoding="utf-8") as f:
        first_char = ""
        while True:
            char = f.read(1)
            if not char:
                break
            if not char.isspace():
                first_char = char
                break

    if first_char == "[":
        return count_json_array(path)

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        return len(data)
    raise ValueError(f"Unsupported JSON root type: {type(data).__name__}")


def count_dataset(path: Path) -> int:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        return count_jsonl(path)
    if suffix == ".json":
        return count_json(path)
    raise ValueError("Only .json and .jsonl files are supported.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quickly count dataset samples.")
    parser.add_argument(
        "dataset_or_path",
        type=str,
        help="Dataset name, e.g. demo, or a path to a .json/.jsonl file.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    dataset_path = resolve_dataset_path(args.dataset_or_path)
    sample_count = count_dataset(dataset_path)
    print(f"{dataset_path}: {sample_count}")
