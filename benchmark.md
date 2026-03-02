# Benchmarks

## Benchmark 1 — Initial run

**Date:** 2026-03-01

### Model

| Field | Value |
|-------|-------|
| Model name | traces1 |
| Model ID | `0088fbdb-04b5-44c3-b549-51015c6be758` |
| Upload ID | `5db79a5a-c960-4d62-aac8-483f78760401` |
| Teacher evaluation ID | `f3e0bbe4-f65c-4ae2-a058-adb7fe50902a` |
| Training ID | `11ca0e6e-a8ae-4538-8d47-87b93c465188` |

### Configuration

| Parameter | Value |
|-----------|-------|
| Task | tool-calling-closed-book |
| Student model | Qwen3-0.6B |
| Teacher model | openai.gpt-oss-120b |
| Random seed | 42 |

### Data

| Split | Rows |
|-------|------|
| Train | 80 |
| Test | 83 |
| Unstructured | 1107 |

- Source: `data/massive_en-US_iot_function_calling.jsonl` (1107 rows, 9 IoT functions)
- Sampled 297 rows (stratified), annotated with LLM quality scores
- Filtered to 163 rows (min inference_score=5, min coherence_score=5)
- Split 50/50 stratified into train/test

### Results

**Status:** JOB_SUCCESS (2026-03-01 19:08 – 21:30, ~2.5 hours)

| Model | Rouge | Tool Call Equivalence |
|-------|-------|----------------------|
| Teacher (openai.gpt-oss-120b) | 87.94% | 50.60% |
| Student base (Qwen3-0.6B) | 50.02% | 9.64% |
| Student tuned (Qwen3-0.6B) | **96.89%** | **78.31%** |
