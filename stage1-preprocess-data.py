"""Preprocess Amazon MASSIVE JSONL data into conversation traces and upload to Hugging Face.

Reads the raw MASSIVE JSONL, converts each row into a conversation trace
(user message + assistant tool call), and uploads the result to Hugging Face
as a Parquet dataset using dlt.

Input format (per line, from the raw tarball):
    {"id": "1", "locale": "en-US", "partition": "train",
     "scenario": "alarm", "intent": "alarm_set",
     "utt": "wake me up at nine am on friday",
     "annot_utt": "wake me up at [time : nine am] on [date : friday]",
     "worker_id": "1"}

Output: Hugging Face dataset at hf://datasets/<namespace>/massive-<scenario>-<timestamp>

Usage:
    python stage1-preprocess-data.py --hf-namespace my-org
    python stage1-preprocess-data.py --hf-namespace my-org --scenario iot
    HF_NAMESPACE=my-org python stage1-preprocess-data.py
"""

import argparse
import os
import re
from datetime import datetime, timezone
from pathlib import Path

import dlt
from dlt.sources.filesystem import filesystem, read_jsonl

SLOT_PATTERN = re.compile(r"\[(\w+)\s*:\s*([^\]]+)\]")


def parse_arguments(annot_utt: str) -> dict:
    """Extract slot key-value pairs from annotated utterance."""
    return {m.group(1): m.group(2).strip() for m in SLOT_PATTERN.finditer(annot_utt)}


def convert_row(row: dict) -> dict:
    """Convert a raw MASSIVE row into a conversation trace."""
    messages = [
        {"role": "user", "content": row["utt"]},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {
                        "name": row["intent"],
                        "arguments": parse_arguments(row["annot_utt"]),
                    },
                }
            ],
        },
    ]
    yield {
        "messages": messages,
        "scenario": row["scenario"],
        "partition": row["partition"],
    }


def main(input_path: Path, scenario: str, hf_namespace: str):
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d-%H-%M-%S")
    dataset_name = f"massive_{scenario}_{timestamp}".replace("-", "_")

    dlt.secrets["bucket_url"] = f"hf://datasets/{hf_namespace}"

    pipeline = dlt.pipeline(
        pipeline_name="massive_traces",
        destination="filesystem",
        dataset_name=dataset_name,
    )

    print(f"Processing scenario '{scenario}' from {input_path}")
    print(f"Uploading to {dlt.secrets.get('bucket_url')}/{dataset_name}")

    resource = (
        filesystem(bucket_url=str(input_path.parent), file_glob=input_path.name)
        | read_jsonl()
    ).with_name("massive_traces")

    resource.apply_hints(write_disposition="replace", columns={"messages": {"data_type": "json"}})
    resource.add_filter(lambda row: row["scenario"] == scenario)
    resource.add_yield_map(convert_row)

    load_info = pipeline.run(resource)

    print(load_info)
    print(f"\nDataset: https://huggingface.co/datasets/{hf_namespace}/{dataset_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/1.1/data/en-US.jsonl"),
        help="Path to the MASSIVE JSONL file",
    )
    parser.add_argument(
        "--scenario",
        default="iot",
        help="Scenario to filter on (default: iot)",
    )
    parser.add_argument(
        "--hf-namespace",
        default=os.environ.get("HF_NAMESPACE", ""),
        help="Hugging Face username or org (or set HF_NAMESPACE env var)",
    )
    args = parser.parse_args()

    if not args.hf_namespace:
        parser.error("--hf-namespace is required (or set HF_NAMESPACE env var)")

    main(args.input, args.scenario, args.hf_namespace)
