import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]


from cluster import cluster_const
from entropy import cal_cohesion_weights  # noqa: E402
from utils import load_json, read_feature, save_json  # noqa: E402


def _resolve_existing_file(path: Path, description: str) -> Path:
    if not path.exists():
        print(f"{description} not found: {path}")
        return None
    return path


def _load_source_data(dataset: str, source: str) -> list[dict]:
    source_path = ROOT_DIR / "data" / ("raw" if source == "raw" else "") / f"{dataset}.json"
    if source == "raw" and not source_path.exists():
        print(f"Raw data not found at {source_path}; falling back to processed data.")
        source_path = ROOT_DIR / "data" / f"{dataset}.json"
    _resolve_existing_file(source_path, "Source data")
    return load_json(str(source_path))


def _feature_db_path(dataset: str) -> Path:
    path = ROOT_DIR / "output" / "feature" / f"{dataset}.db"
    if path.exists():
        return path

    normalized_path = ROOT_DIR / "output" / "feature" / f"{dataset.replace('-', '_')}.db"
    return _resolve_existing_file(normalized_path, "Feature database")


def build_scs_selection(
    source_data: list[dict],
    features: dict,
    clusters: list[dict],
    sample_size: int,
) -> tuple[list[dict], list[dict]]:
    if sample_size <= 0:
        raise ValueError("sample_size must be positive.")
    if len(source_data) < len(clusters):
        raise ValueError(
            f"Source data has {len(source_data)} records, but clusters contain {len(clusters)} records."
        )

    sem_ids = [cluster["sem_id"] for cluster in clusters]
    embeds = [features[cluster["doc_id"]]["embedding"] for cluster in clusters]
    cohesion_weights = cal_cohesion_weights(sem_ids, embeds)

    cluster_to_items = defaultdict(list)
    for cluster, cohesion_weight in zip(clusters, cohesion_weights):
        item = {
            "doc_id": cluster["doc_id"],
            "sem_id": cluster["sem_id"],
            "cohesion_weight": cohesion_weight,
        }
        cluster_to_items[cluster["sem_id"]].append(item)

    for sem_id in cluster_to_items:
        cluster_to_items[sem_id].sort(
            key=lambda item: (-item["cohesion_weight"], item["doc_id"])
        )

    ordered_sem_ids = sorted(cluster_to_items)
    cursors = {sem_id: 0 for sem_id in ordered_sem_ids}
    selected_items = []
    target_size = min(sample_size, len(clusters))

    while len(selected_items) < target_size:
        added_this_round = False
        for sem_id in ordered_sem_ids:
            cursor = cursors[sem_id]
            items = cluster_to_items[sem_id]
            if cursor >= len(items):
                continue
            selected_items.append(items[cursor])
            cursors[sem_id] += 1
            added_this_round = True
            if len(selected_items) >= target_size:
                break
        if not added_this_round:
            break

    selected_data = [source_data[item["doc_id"]] for item in selected_items]
    return selected_data, selected_items


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Select data by SCS: rank samples inside each semantic cluster by Semantic "
            "Cohesion Weight, then round-robin across clusters."
        )
    )
    parser.add_argument("--dataset", type=str, help="Dataset name, e.g. demo")
    parser.add_argument(
        "--sample_size",
        type=int,
        required=True,
        help="Number of samples to select.",
    )
    parser.add_argument(
        "--source",
        choices=["raw", "processed"],
        default="raw",
        help="Write selected records from data/raw/<dataset>.json or data/<dataset>.json.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON path. Defaults to output/selection/<dataset>_scs_<sample_size>.json.",
    )
    parser.add_argument(
        "--metadata_output",
        type=str,
        default=None,
        help="Optional path for selected doc_id, sem_id, and cohesion_weight metadata.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    source_data = _load_source_data(args.dataset, args.source)
    feature_path = _feature_db_path(args.dataset)
    cluster_path = _resolve_existing_file(
        ROOT_DIR / "output" / "cluster" / f"{args.dataset}.json",
        "Cluster file",
    )
    if not cluster_path:
        cluster_const(args.dataset)
        cluster_path = _resolve_existing_file(
        ROOT_DIR / "output" / "cluster" / f"{args.dataset}.json",
        "Cluster file",
    )
    features = read_feature(str(feature_path).replace("\\", "/"))
    clusters = load_json(str(cluster_path))
    selected_data, selected_items = build_scs_selection(
        source_data=source_data,
        features=features,
        clusters=clusters,
        sample_size=args.sample_size,
    )

    output_path = (
        Path(args.output)
        if args.output
        else ROOT_DIR / "output" / "selection" / f"{args.dataset}_SCS_{args.sample_size}.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(selected_data, str(output_path))

    if args.metadata_output:
        metadata_path = Path(args.metadata_output)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        save_json(selected_items, str(metadata_path))

    selected_clusters = sorted({item["sem_id"] for item in selected_items})
    print(
        json.dumps(
            {
                "dataset": args.dataset,
                "selected": len(selected_data),
                "requested": args.sample_size,
                "cluster_count": len(selected_clusters),
                "output": str(output_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
