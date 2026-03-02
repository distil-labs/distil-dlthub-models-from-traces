# Training a Small Language Model from Production Traces

Your production LLM agent is already solving the task thousands of times a day. Those traces aren't just logs -- they're a description of your problem space: the vocabulary your users actually use, the edge cases that show up in the wild, the distribution of requests your model needs to handle. This project shows how to turn that signal into a purpose-built small model that replaces your large one.

This demo chains three tools into a complete ML pipeline:

- **[dlt](https://dlthub.com)** extracts and preprocesses your production traces from wherever they live -- any database, API, cloud storage, or log aggregator -- and writes the cleaned dataset to Hugging Face.
- **[Hugging Face](https://huggingface.co)** acts as the shared data and model hub: the cleaned trace dataset is stored there, and the trained model is published there for deployment.
- **[Distil Labs](https://distillabs.ai)** reads the trace dataset from Hugging Face, uses it as domain context to generate high-quality synthetic training data, fine-tunes a small student model, and pushes the result back to Hugging Face.

The key thing to understand about the Distil Labs step: we are not training directly on your traces. We use them as context to steer synthetic data generation with a large teacher model -- one that now understands *your* domain, *your* phrasing patterns, and *your* edge cases, not just the general case. The result is a model that generalizes correctly to real traffic, not just to held-out benchmark examples.

As a concrete example, we train a **0.6B parameter model** that routes IoT smart home commands to the correct function call.

## Results

Training a **Qwen3-0.6B** student distilled from an **openai.gpt-oss-120b** teacher, using traces extracted by dlt, stored on Hugging Face, and used by Distil Labs to ground synthetic data generation:

| Model | Tool Call Equivalence (↑) | Parameters |
|-------|--------------------------|------------|
| Teacher (openai.gpt-oss-120b) | 50.60% | 120B |
| Student base (Qwen3-0.6B) | 9.64% | 0.6B |
| **Student tuned (Qwen3-0.6B)** | **78.31%** | **0.6B** |

The tuned 0.6B model **beats the 120B teacher by 28 points** on exact structured match. The teacher scores lower here because it is a general-purpose model -- it has never specialized in your domain, your function schemas, or the specific phrasing patterns of your users. The student, trained on synthetic data generated from real traffic, is an expert in exactly this task and nothing else. A model 200x smaller, running in under 50ms locally versus 400-700ms for a cloud API call, doing the job better on the metric that matters for production: did it call the right function with the right arguments?

---

## The Pipeline

```
Production traces                 Hugging Face                    Hugging Face
(any source)                      (data hub)                      (model hub)
     │                                │                                │
     ▼                                ▼                                ▼
┌─────────┐   cleaned traces   ┌─────────────┐  traces + seed  ┌─────────────┐
│   dlt   │ ─────────────────► │  HF Dataset │ ───────────────► │ Distil Labs │
└─────────┘                    └─────────────┘                  └──────┬──────┘
                                                                        │
                                                         synthetic data │ generation
                                                         + fine-tuning  │
                                                                        ▼
                                                               ┌─────────────┐
                                                               │  HF Model   │
                                                               └─────────────┘
```

**dlt** handles the messy part: connecting to your production data store (Postgres, S3, BigQuery, log aggregators) and writing clean, structured traces to Hugging Face. This step is source-agnostic -- you swap the dlt connector, not the rest of the pipeline.

**Hugging Face** acts as the shared hub between tools. The cleaned trace dataset lands there after Stage 1, and the trained model is published there after training. Both are versioned, shareable, and accessible to the rest of your stack.

**Distil Labs** reads the trace dataset from Hugging Face, uses it as domain context to generate ~10,000 synthetic training examples with a large teacher model, fine-tunes a compact student, and pushes the result back to Hugging Face. The traces tell the teacher what your domain looks like; the teacher writes the actual training data.

---

## Prerequisites

```bash
# Install dependencies
pip install dlt[filesystem,hf] litellm datasets

# Install Distil Labs CLI
curl -fsSL https://cli-assets.distillabs.ai/install.sh | sh
distil login

# Download the raw data
mkdir -p data
curl -o data/amazon-massive-dataset-1.1.tar.gz \
  https://amazon-massive-nlu-dataset.s3.amazonaws.com/amazon-massive-dataset-1.1.tar.gz
tar -xzf data/amazon-massive-dataset-1.1.tar.gz -C data/
```

We use the [Amazon MASSIVE](https://huggingface.co/datasets/AmazonScience/massive) dataset as a stand-in for production traffic. It contains 16k+ natural language utterances across 60 intents and 18 scenarios. We use the IoT scenario (smart home commands like "turn on the kitchen lights" or "make me a coffee at 7am") as our target agent's traffic.

---

## Stage 1: Extract and Upload Production Traces with dlt

**Script:** `stage1-preprocess-data.py`

In a real system, production traces are scattered across databases, cloud storage, and log aggregators -- mixed in with traffic from multiple agents and stored in whatever format each service uses. [dlt](https://dlthub.com) solves this: it can [load data from any source](https://dlthub.com/docs/dlt-ecosystem/verified-sources/) -- Postgres, Snowflake, S3, BigQuery, local files, REST APIs -- and write it to a consistent destination. The same pipeline logic works regardless of where your traces live; you change the source connector, not the transformation or upload code.

The destination here is Hugging Face. dlt writes the cleaned traces as a versioned Parquet dataset using the [filesystem destination](https://dlthub.com/docs/dlt-ecosystem/destinations/filesystem), making them immediately available to Distil Labs in the next stage.

Here we load from a local file as a stand-in for a production data store. The pipeline:

1. **Filters** -- selects only traces relevant to our target agent (the `iot` scenario, covering 9 smart home functions like `iot_hue_lighton`, `iot_wemo_off`, `iot_coffee`, etc.)
2. **Formats** -- converts each raw record into a conversation trace (a user message paired with an assistant tool call) in the OpenAI function-calling format
3. **Uploads** -- writes the processed traces to Hugging Face as a Parquet dataset using dlt's [filesystem destination](https://dlthub.com/docs/dlt-ecosystem/destinations/filesystem)

```bash
python stage1-preprocess-data.py --hf-namespace my-org
```

This produces a dataset like `my-org/massive_iot_2026_03_02_09_16_28` on Hugging Face containing 1,107 IoT conversation traces.

---

## Stage 2: Curate Training Data for Distil Labs

**Script:** `stage2-prepare-distil-labs-data.py`

Distil Labs only needs a small number of clean, representative examples as a seed -- the platform generates the bulk of the training data synthetically using a large teacher model. The data preparation work here is minimal: filter your traces down to a few dozen high-quality examples. This stage does that automatically using an LLM as judge, so you don't need to review examples by hand.

1. **Loads** the conversation traces from Hugging Face and the function schemas from `job_description.json`
2. **Samples** ~300 rows using stratified sampling across all 9 functions
3. **Scores** each sample with LLM quality scores (inference clarity and utterance coherence, 1-5 scale)
4. **Filters** to keep only examples where both scores are perfect -- clear intent and natural phrasing
5. **Splits** the filtered examples 50/50 into stratified train/test sets
6. **Writes** the Distil Labs upload files: `train.jsonl`, `test.jsonl`, and `unstructured.jsonl`

The most important output is `unstructured.jsonl`, which contains all 1,107 original production traces. This is what grounds the synthetic data generation: the Distil Labs pipeline feeds these traces to the teacher model as domain context, so the generated training examples reflect the vocabulary, phrasing, and edge cases of your real traffic rather than the model's generic priors.

```bash
python stage2-prepare-distil-labs-data.py --input my-org/massive_iot_2026_03_02_09_16_28
```

This produces the `finetuning-data/` directory ready for upload:

| File | Description |
|------|-------------|
| `job_description.json` | Task description and tool schemas (9 IoT functions) |
| `config.yaml` | Training configuration (task type, student/teacher models) |
| `train.jsonl` | ~75 filtered seed examples |
| `test.jsonl` | ~78 held-out evaluation examples |
| `unstructured.jsonl` | 1,107 production traces used to ground synthetic data generation |

Upload the data and kick off training with the Distil CLI:

```bash
distil model create my-iot-model
distil model upload-data <model-id> --data ./finetuning-data
distil model run-teacher-evaluation <model-id>
distil model run-training <model-id>
```

What happens next: the Distil Labs pipeline uses your seed examples and unstructured traces to prompt the teacher model to generate ~10,000 synthetic training examples in your domain. Each generated example is validated and filtered before entering the training set. The student model is then fine-tuned on this curated synthetic dataset -- not directly on your traces.

---

## Stage 3: Deploy the Trained Model from Hugging Face

### Upload to Hugging Face

Push the trained model to a Hugging Face repository using the [Distil Labs API](https://docs.distillabs.ai/api-reference/trainings/upload-huggingface-model-trainings-training-id-huggingface-models-post). This requires a Distil Labs API token (see [Appendix: Getting a Distil Labs API Token](#appendix-getting-a-distil-labs-api-token)). The model trained in this tutorial is available at [distillabs/massive-iot-traces1](https://huggingface.co/distillabs/massive-iot-traces1).

```bash
curl -X POST "https://api.distillabs.ai/trainings/<training-id>/huggingface_models" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $DISTIL_API_TOKEN" \
  -d '{"hf_token": "<your-hf-token>", "repo_id": "<your-org/model-name>"}'
```

### Remote Deployment

Deploy the model to Distil Labs managed infrastructure:

```bash
distil model deploy remote <model-id>
```

This provisions a vLLM-based endpoint and returns the URL and API key. Deployment takes a few minutes to complete.

### Test the Deployment

Once deployed, get the client script to test the model:

```bash
distil model invoke <model-id>
```

Example responses from the deployed model:

```
"turn on the kitchen lights"     ->  iot_hue_lighton(house_place="kitchen")
"make me a coffee at 7am"        ->  iot_coffee(time="7am")
"dim the bedroom lights"         ->  iot_hue_lightdim(house_place="bedroom")
"turn off the smart plug"        ->  iot_wemo_off(device_type="smart plug")
```

### Deactivate Deployment

When you're done testing, deactivate the deployment to stop consuming inference credits:

```bash
distil model deploy remote --deactivate <model-id>
```

### Local Deployment

For development and testing, you can also deploy the model locally using llama.cpp:

```bash
distil model deploy local <model-id>
```

---

## Model Variants

| Model | Format | Size | Use Case |
|-------|--------|------|----------|
| [distillabs/massive-iot-traces1](https://huggingface.co/distillabs/massive-iot-traces1) | Safetensors (BF16) | ~1.2 GB | Transformers, vLLM, cloud deployment |
| GGUF quantized variants | Coming soon | -- | Ollama, llama.cpp, local inference |

For local development, watch the HuggingFace repo for GGUF variants. For production inference, use the Distil Labs remote deployment or vLLM directly with the Safetensors model.

---

## Project Structure

```
.
├── README.md                          # This file
├── stage1-preprocess-data.py          # Extract & upload production traces
├── stage2-prepare-distil-labs-data.py # Curate training data & prepare upload
├── finetuning-data/
│   ├── job_description.json           # Task + tool schemas
│   ├── config.yaml                    # Training configuration
│   ├── train.jsonl                    # Seed training examples
│   ├── test.jsonl                     # Held-out evaluation set
│   └── unstructured.jsonl             # Production traces for synthetic data context
├── benchmark.md                       # Training run details and results
└── data/                              # Raw MASSIVE dataset (gitignored)
```

---

## Use This Pipeline for Your Own Agent

The dlt + Hugging Face + Distil Labs pipeline generalizes to any agent that handles a bounded, well-defined task: function routing, intent classification, PII redaction, structured extraction, and so on. If you have production traces and a working LLM agent, you have everything you need.

1. **Export your traces with dlt.** Swap the source connector in `stage1` for whatever holds your logs -- Postgres, S3, BigQuery, a REST API. The rest of the pipeline is unchanged.
2. **Define your task format.** Write a `job_description.json` describing the input/output format and your tool schemas or output classes.
3. **Filter a small seed set.** Use the LLM scoring in `stage2` to automatically identify ~50-100 clean, representative examples from your traces. No manual annotation required.
4. **Upload traces and seed data to Hugging Face.** `stage2` writes the files; the Distil Labs CLI uploads them. Your trace dataset is now versioned and accessible.
5. **Run the Distil Labs pipeline.** Distil Labs reads your dataset from Hugging Face, generates synthetic training data grounded in your traces, fine-tunes the student, and publishes the model back to Hugging Face. Training completes in under 12 hours. No ML expertise required.

Visit [distillabs.ai](https://www.distillabs.ai/) to get started, or [dlthub.com](https://dlthub.com) to explore source connectors for your data store.

---

## FAQ

**Q: Why not just use GPT-4 / Claude for this in production?**

Two reasons: cost and latency. A 0.6B model deployed on a single GPU responds in under 50ms. A cloud API call takes 400-700ms and costs roughly 200x more per inference at scale. For agents handling thousands of requests per day, that gap compounds fast. And if your data is sensitive -- as it often is for IoT, fintech, or healthcare agents -- you may not be able to send it to a third-party API at all.

**Q: Why not just prompt the base Qwen3-0.6B model directly?**

The base model achieves 9.64% tool call equivalence on this task -- it knows roughly what a function call looks like, but it doesn't know your specific functions, argument names, or the phrasing patterns of your users. Fine-tuning on synthetic data grounded in your traces is what closes that gap.

**Q: Are you just training on my traces directly?**

No. Your traces are used as domain context to steer synthetic data generation -- the teacher model reads them to understand your problem space and produces ~10,000 purpose-built training examples. The student is trained on that synthetic dataset, not on raw traces. This matters because raw traces from a production LLM often contain formatting inconsistencies that make poor direct training signal.

**Q: Do I need to label all my production traces?**

No. You only need a small filtered seed set (~50-100 examples). The LLM scoring step in `stage2` handles this automatically -- it scores each candidate on inference clarity and utterance coherence and keeps only the cleanest examples. The rest of your traces go in as unlabeled context.

**Q: The model produces an incorrect function call.**

The model achieves 78.31% exact match, which means roughly 1 in 5 queries may need a fallback or review step. For production use, consider adding a confidence threshold and routing low-confidence predictions to a larger model. If you find consistent failure patterns, open an issue and we can look at adding those cases to the training data.

---

## Appendix: Getting a Distil Labs API Token

The Distil Labs API requires a bearer token for authentication. Tokens are valid for 1 hour. See the [authentication docs](https://docs.distillabs.ai/getting-started/authentication#api-authentication) for the recommended approach. As a quick alternative:

```bash
export DISTIL_API_TOKEN=$(curl -s -X POST "https://cognito-idp.eu-central-1.amazonaws.com" \
  -H "X-Amz-Target: AWSCognitoIdentityProviderService.InitiateAuth" \
  -H "Content-Type: application/x-amz-json-1.1" \
  -d '{
    "AuthFlow": "USER_PASSWORD_AUTH",
    "ClientId": "4569nvlkn8dm0iedo54nbta6fd",
    "AuthParameters": {
      "USERNAME": "<your-email>",
      "PASSWORD": "<your-password>"
    }
  }' | python3 -c "import sys,json; print(json.load(sys.stdin)['AuthenticationResult']['AccessToken'])")
```

---

## Links

**Partners**

[![dlt](https://img.shields.io/badge/dlt-Homepage-green)](https://dlthub.com/)
[![dlt Docs](https://img.shields.io/badge/dlt-Docs-green)](https://dlthub.com/docs)
[![Hugging Face](https://img.shields.io/badge/HuggingFace-distillabs-yellow)](https://huggingface.co/distillabs)
[![Distil Labs](https://img.shields.io/badge/Distil_Labs-Homepage-blue)](https://www.distillabs.ai/)
[![Distil Labs Docs](https://img.shields.io/badge/Distil_Labs-Docs-blue)](https://docs.distillabs.ai/)

**Community**

[![GitHub](https://img.shields.io/badge/GitHub-distil--labs-black)](https://github.com/distil-labs)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Distil_Labs-blue)](https://www.linkedin.com/company/distil-labs/)
[![Slack](https://img.shields.io/badge/Slack-Community-purple)](https://distil-labs-community.slack.com)
