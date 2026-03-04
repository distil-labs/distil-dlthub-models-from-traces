"""Prepare the Distil Labs finetuning upload from a Hugging Face conversation trace dataset.

Pipeline:
  1. Load inputs     – Read HF dataset and function schema from job_description.json
  2. Sample          – Stratified sample ~N rows balanced across functions
  3. Annotate        – LLM quality scoring (inference_score, coherence_score)
  4. Filter          – Keep only rows where both scores >= --min-score
  5. Split           – Stratified train/test split into equal halves
  6. Write outputs   – train.jsonl, test.jsonl, unstructured.jsonl to --output-dir

Usage:
    python stage2-prepare-distil-labs-data.py --input my-org/massive-iot-2026-03-01
    python stage2-prepare-distil-labs-data.py --input my-org/massive-iot-2026-03-01 --min-score 4
    python stage2-prepare-distil-labs-data.py --input my-org/massive-iot-2026-03-01 --model openai/gpt-4o
"""

import argparse
import json
import random
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import dlt
import litellm

# ── Constants ────────────────────────────────────────────────────────

SAMPLE_SIZE = 300
MIN_SCORE = 5
TEST_FRACTION = 0.5
SEED = 42
DEFAULT_MODEL = "bedrock/converse/openai.gpt-oss-120b-1:0"

ANNOTATION_PROMPT = """\
You are evaluating a function-calling dataset for quality. Given a user utterance \
and the function call it was mapped to, rate the example on two dimensions.

## Function call
- Name: {func_name}
- Description: {func_description}
- Arguments: {arguments}

## User utterance
"{utterance}"

## Rating instructions
Rate each dimension from 1 to 5:

**inference_score**: Can the correct function and its arguments be clearly inferred \
from the utterance alone?
  1 = Impossible to infer
  2 = Very ambiguous
  3 = Somewhat inferable
  4 = Mostly clear
  5 = Completely obvious

**coherence_score**: Does the utterance make sense as something a real user would say?
  1 = Nonsensical / garbled
  2 = Barely understandable
  3 = Understandable but awkward
  4 = Natural with minor issues
  5 = Perfectly natural

Respond with ONLY a JSON object, no other text:
{{"inference_score": <int>, "coherence_score": <int>}}"""


# ── Helpers ──────────────────────────────────────────────────────────


def normalize_row(row: dict) -> dict:
    """Extract utterance and function_call from conversation trace format.

    Handles messages as either a list or a JSON string (HF Parquet).
    """
    messages = row["messages"]
    if isinstance(messages, str):
        messages = json.loads(messages)
        row["messages"] = messages
    user_msg = messages[0]
    assistant_msg = messages[1]
    fc = assistant_msg["tool_calls"][0]["function"]
    row["utterance"] = user_msg["content"]
    row["function_call"] = {"name": fc["name"], "arguments": fc["arguments"]}
    return row


def parse_arguments(fc: dict) -> dict:
    """Parse function_call arguments, handling both str and dict."""
    args = fc["arguments"]
    return json.loads(args) if isinstance(args, str) else args


# ── Stage functions ──────────────────────────────────────────────────


def load_inputs(input_dataset: str) -> list[dict]:
    """Stage 1: Load the function schema and input dataset from Hugging Face."""
    print("Stage 1: Load inputs")

    hf_namespace, dataset_name = input_dataset.split("/", 1)
    dlt.secrets["bucket_url"] = f"hf://datasets/{hf_namespace}"
    p = dlt.pipeline(
        destination="filesystem",
        dataset_name=dataset_name,
    )
    all_rows = [normalize_row(row) for row in p.dataset().massive_traces.df().to_dict("records")]
    print(f"  Input:  {len(all_rows)} examples from {input_dataset}")

    return all_rows


def stratified_sample(all_rows: list[dict], rng: random.Random) -> list[dict]:
    """Stage 2: Stratified sample balanced across function names."""
    print("Stage 2: Stratified sample")

    groups: dict[str, list] = defaultdict(list)
    for row in all_rows:
        groups[row["function_call"]["name"]].append(row)

    quota = SAMPLE_SIZE // len(groups)
    sampled: list[dict] = []
    for fn, members in sorted(groups.items()):
        sampled.extend(rng.sample(members, min(quota, len(members))))

    rng.shuffle(sampled)
    print(f"  {len(sampled)} rows sampled")
    return sampled


def get_quality_scores(row: dict, schema: dict, model: str) -> dict:
    """Call the LLM to score a single example. Returns a dict with the scores."""
    fc = row["function_call"]
    args = parse_arguments(fc)
    prompt = ANNOTATION_PROMPT.format(
        func_name=fc["name"],
        func_description=schema[fc["name"]]["description"],
        arguments=json.dumps(args),
        utterance=row["utterance"],
    )
    for attempt in range(3):
        try:
            resp = litellm.completion(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=256,
            )
            result = resp.choices[0].message.content.strip()
            # Strip markdown code fences (e.g. ```json ... ```)
            if result.startswith("```"):
                result = result.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            scores = json.loads(result)
            return {
                "inference_score": int(scores["inference_score"]),
                "coherence_score": int(scores["coherence_score"]),
            }
        except Exception as e:
            if attempt < 2:
                time.sleep(2**attempt)
            else:
                print(f"  FAILED: {row['utterance'][:50]}... — {e}")
                return {"inference_score": -1, "coherence_score": -1}


def annotate_rows(sampled: list[dict], schema: dict, model: str) -> list[dict]:
    """Stage 3: Annotate each row with LLM quality scores."""
    print(f"Stage 3: Annotate ({len(sampled)} rows with {model})")

    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(get_quality_scores, row=row, schema=schema, model=model)
            for row in sampled
        ]
        results = [future.result() for future in futures]

    for row, scores in zip(sampled, results):
        row.update(scores)
    print(f"  {len(sampled)}/{len(sampled)} done")

    return sampled


def filter_by_quality(sampled: list[dict], min_score: int) -> list[dict]:
    """Stage 4: Keep only rows where both scores >= min_score."""
    print(f"Stage 4: Filter (min_score={min_score})")

    filtered = [
        r for r in sampled
        if r.get("inference_score", 0) >= min_score
        and r.get("coherence_score", 0) >= min_score
    ]
    print(f"  {len(filtered)} of {len(sampled)} rows kept")
    return filtered


def train_test_split(
    filtered: list[dict], rng: random.Random
) -> tuple[list[dict], list[dict]]:
    """Stage 5: Stratified train/test split by function name."""
    print(f"Stage 5: Train/test split (test_fraction={TEST_FRACTION})")

    groups: dict[str, list] = defaultdict(list)
    for row in filtered:
        groups[row["function_call"]["name"]].append(row)

    train_rows, test_rows = [], []
    for fn in sorted(groups):
        members = list(groups[fn])
        rng.shuffle(members)
        n_test = max(1, round(len(members) * TEST_FRACTION))
        test_rows.extend(members[:n_test])
        train_rows.extend(members[n_test:])

    rng.shuffle(train_rows)
    rng.shuffle(test_rows)
    print(f"  Train: {len(train_rows)}, Test: {len(test_rows)}")
    return train_rows, test_rows


def write_outputs(
    train_rows: list[dict], test_rows: list[dict], all_rows: list[dict],
    output_dir: Path,
) -> None:
    """Stage 6: Write train.jsonl, test.jsonl, and unstructured.jsonl."""
    print("Stage 6: Write output files")
    output_dir.mkdir(parents=True, exist_ok=True)

    def to_distil_row(row: dict) -> dict:
        fc = row["function_call"]
        args = parse_arguments(fc)
        return {
            "question": row["utterance"],
            "answer": json.dumps({"name": fc["name"], "parameters": args}),
        }

    for name, rows in [("train.jsonl", train_rows), ("test.jsonl", test_rows)]:
        path = output_dir / name
        with open(path, "w") as f:
            for row in rows:
                f.write(json.dumps(to_distil_row(row)) + "\n")
        print(f"  {path} ({len(rows)} rows)")

    unstructured_path = output_dir / "unstructured.jsonl"
    with open(unstructured_path, "w") as f:
        for row in all_rows:
            f.write(json.dumps({"context": json.dumps(row["messages"])}) + "\n")
    print(f"  {unstructured_path} ({len(all_rows)} rows)")


# ── Main ─────────────────────────────────────────────────────────────


def main(input_dataset: str, job_description: Path, output_dir: Path, seed: int = SEED, model: str = DEFAULT_MODEL):
    rng = random.Random(seed)

    with open(job_description) as f:
        job_desc = json.load(f)
    schema = {t["function"]["name"]: t["function"] for t in job_desc["tools"]}
    print(f"  Schema: {len(schema)} functions from {job_description}")

    all_rows = load_inputs(input_dataset)
    sampled = stratified_sample(all_rows, rng)
    annotated = annotate_rows(sampled, schema, model)
    filtered = filter_by_quality(annotated, MIN_SCORE)
    train_rows, test_rows = train_test_split(filtered, rng)
    sampled_ids = {id(r) for r in sampled}
    unstructured_rows = [r for r in all_rows if id(r) not in sampled_ids]
    write_outputs(train_rows, test_rows, unstructured_rows, output_dir)

    print(f"\nDone. Upload directory: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--input", required=True,
        help="Hugging Face dataset ID (e.g. my-org/massive-iot-2026-03-01-12-00-00)",
    )
    parser.add_argument(
        "--job-description", type=Path, default=Path("finetuning-data/job_description.json"),
        help="Path to Distil Labs job_description.json",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("finetuning-data"),
        help="Directory to write train.jsonl, test.jsonl, and unstructured.jsonl",
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL,
        help="LLM model for annotation in litellm format (e.g. openai/gpt-4o, bedrock/converse/...)",
    )
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    main(args.input, args.job_description, args.output_dir, args.seed, args.model)
