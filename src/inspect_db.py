import argparse
import pickle
import sqlite3
from pathlib import Path
from typing import Any


def summarize_value(value: Any, max_text: int) -> Any:
    if value is None or isinstance(value, (int, float)):
        return value

    if isinstance(value, str):
        return value if len(value) <= max_text else value[:max_text] + "..."

    if isinstance(value, bytes):
        summary = {
            "type": "BLOB",
            "bytes": len(value),
        }
        try:
            obj = pickle.loads(value)
        except Exception as exc:
            summary["pickle"] = f"unreadable: {type(exc).__name__}: {exc}"
            return summary

        summary["pickle_type"] = type(obj).__name__
        if hasattr(obj, "shape"):
            summary["shape"] = tuple(obj.shape)
            summary["dtype"] = str(getattr(obj, "dtype", "unknown"))
        elif isinstance(obj, (list, tuple)):
            summary["length"] = len(obj)
            summary["preview"] = list(obj[:5])
        elif isinstance(obj, dict):
            summary["keys"] = list(obj.keys())[:10]
        else:
            text = repr(obj)
            summary["repr"] = text if len(text) <= max_text else text[:max_text] + "..."
        return summary

    text = repr(value)
    return text if len(text) <= max_text else text[:max_text] + "..."


def quote_identifier(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def inspect_db(db_path: Path, sample_rows: int, max_text: int) -> None:
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT name, type, sql
        FROM sqlite_master
        WHERE type IN ('table', 'view')
        ORDER BY type, name
        """
    )
    objects = cursor.fetchall()

    print(f"Database: {db_path}")
    print(f"Objects: {len(objects)}")

    for obj in objects:
        name = obj["name"]
        kind = obj["type"]
        quoted_name = quote_identifier(name)

        print("\n" + "=" * 80)
        print(f"{kind.upper()}: {name}")
        print("-" * 80)
        print(obj["sql"] or "<no create sql>")

        cursor.execute(f"PRAGMA table_info({quoted_name})")
        columns = cursor.fetchall()
        print("\nColumns:")
        for col in columns:
            print(
                "  "
                f"cid={col['cid']} "
                f"name={col['name']} "
                f"type={col['type']} "
                f"notnull={col['notnull']} "
                f"default={col['dflt_value']} "
                f"pk={col['pk']}"
            )

        try:
            cursor.execute(f"SELECT COUNT(*) AS n FROM {quoted_name}")
            row_count = cursor.fetchone()["n"]
            print(f"\nRow count: {row_count}")
        except sqlite3.DatabaseError as exc:
            print(f"\nRow count: unreadable: {type(exc).__name__}: {exc}")
            continue

        if sample_rows <= 0:
            continue

        cursor.execute(f"SELECT * FROM {quoted_name} LIMIT ?", (sample_rows,))
        rows = cursor.fetchall()
        print(f"\nSample rows: {len(rows)}")
        for index, row in enumerate(rows):
            print(f"  [{index}]")
            for key in row.keys():
                value = summarize_value(row[key], max_text=max_text)
                print(f"    {key}: {value}")

    conn.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect SQLite database schema, columns, row counts, and sample metadata."
    )
    parser.add_argument("db_path", type=Path, help="Path to the .db file.")
    parser.add_argument(
        "--sample_rows",
        type=int,
        default=3,
        help="Number of sample rows to print per table or view.",
    )
    parser.add_argument(
        "--max_text",
        type=int,
        default=200,
        help="Maximum characters to print for long text values.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    inspect_db(args.db_path, sample_rows=args.sample_rows, max_text=args.max_text)
