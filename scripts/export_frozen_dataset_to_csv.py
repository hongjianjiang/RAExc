#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "src" / "dataset"


def load_split(payload, key):
    if key in payload:
        return payload[key]
    return {"pos": payload["pos"], "neg": payload["neg"]}


def iter_rows(split):
    for label_key, label_value in (("pos", 1), ("neg", 0)):
        for word in split[label_key]:
            yield list(word) + [label_value]


def write_csv(path, split):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        for row in iter_rows(split):
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Export a frozen JSON benchmark dataset into the CSV layout expected "
            "by src/neural_networks/{lstm,transformer_encoder}.py."
        )
    )
    parser.add_argument("--dataset_json", type=Path, required=True)
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Dataset stem for <name>.csv and test_<name>.csv. Defaults to the JSON stem.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Destination directory for the neural-network CSV files.",
    )
    args = parser.parse_args()

    with open(args.dataset_json) as f:
        payload = json.load(f)

    name = args.name or args.dataset_json.stem
    train_split = load_split(payload, "train")
    eval_split = load_split(payload, "eval")

    train_csv = args.output_dir / f"{name}.csv"
    test_csv = args.output_dir / f"test_{name}.csv"
    write_csv(train_csv, train_split)
    write_csv(test_csv, eval_split)

    summary = {
        "dataset_json": str(args.dataset_json),
        "train_csv": str(train_csv),
        "test_csv": str(test_csv),
        "train_pos": len(train_split["pos"]),
        "train_neg": len(train_split["neg"]),
        "test_pos": len(eval_split["pos"]),
        "test_neg": len(eval_split["neg"]),
        "max_len": payload.get("max_len"),
        "task": payload.get("task"),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
