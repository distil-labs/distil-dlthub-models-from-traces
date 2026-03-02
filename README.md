# Training a Small Language Model from Production Traces

Large language models are powerful but expensive to run at scale. If your production system already handles thousands of requests through an LLM-powered agent, you're sitting on a goldmine of training data — the traces of every user request and every action your agent took. This project shows how to use those traces to train a small, fast, task-specific model that can replace the large one for a fraction of the cost.

We walk through a three-stage pipeline: extracting and cleaning production traces with [dlt](https://dlthub.com), curating high-quality training data for [Distil Labs](https://distillabs.ai), and deploying the resulting model. As a concrete example, we train a **0.6B parameter model** that routes IoT smart home commands to the correct function — and it ends up **outperforming the 120B teacher** it learned from.

We use the [Amazon MASSIVE](https://huggingface.co/datasets/AmazonScience/massive) dataset as a stand-in for production traffic. It contains 16k+ natural language utterances across 60 intents and 18 scenarios — we treat the IoT scenario (smart home commands like "turn on the kitchen lights" or "make me a coffee at 7am") as our target agent's traffic.

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

## Stage 1: Extract and Upload Production Traces

**Script:** `stage1-preprocess-data.py`

The first challenge is getting your production traces into a clean, usable format. In a real system, traces are scattered across databases, cloud storage, and log aggregators — mixed in with traffic from other agents and stored in whatever format each service uses. This stage uses [dlt](https://dlthub.com) to build a data pipeline that pulls raw traces from their source, transforms them, and writes them to a destination. Since dlt can [load data from any source](https://dlthub.com/docs/dlt-ecosystem/verified-sources/) — databases, APIs, cloud storage, local files — the same approach works regardless of where your traces live.

Here we load from a local file, but in practice you'd point dlt at your production data store. The pipeline:

1. **Filters** — selects only traces relevant to our target agent (the `iot` scenario, covering 9 smart home functions like `iot_hue_lighton`, `iot_wemo_off`, `iot_coffee`, etc.)
2. **Formats** — converts each raw record into a conversation trace (a user message paired with an assistant tool call) in the OpenAI function-calling format
3. **Uploads** — writes the processed traces to Hugging Face as a Parquet dataset using dlt's [filesystem destination](https://dlthub.com/docs/dlt-ecosystem/destinations/filesystem)

```bash
python stage1-preprocess-data.py --hf-namespace my-org
```

This produces a dataset like `my-org/massive_iot_2026_03_02_09_16_28` on Hugging Face containing 1,107 IoT conversation traces.

## Stage 2: Curate Training Data and Prepare Upload

**Script:** `stage2-prepare-distil-labs-data.py`

With clean traces in hand, the next step is preparing the training data for Distil Labs. The good news: Distil Labs needs very little labeled data — as few as 20 examples — because it uses knowledge distillation to generate synthetic training data from a large teacher model. The key is making sure those few examples are high quality. This stage automates the curation:

1. **Loads** the conversation traces from Hugging Face and the function schemas from `job_description.json`
2. **Samples** ~300 rows using stratified sampling across all 9 functions
3. **Annotates** each sample with LLM quality scores (inference clarity and utterance coherence, 1–5 scale) using a large model as judge
4. **Filters** to keep only examples where both scores are perfect — clear intent and natural phrasing
5. **Splits** the filtered examples 50/50 into stratified train/test sets
6. **Writes** the Distil Labs upload files: `train.jsonl`, `test.jsonl`, and `unstructured.jsonl`

The unstructured file contains all 1,107 original production traces. Distil Labs uses these as additional context during synthetic data generation, helping the teacher model produce training examples that match the style and distribution of real traffic.

```bash
python stage2-prepare-distil-labs-data.py --input my-org/massive_iot_2026_03_02_09_16_28
```

This produces the `finetuning-data/` directory ready for upload:

| File | Description |
|------|-------------|
| `job_description.json` | Task description and tool schemas (9 IoT functions) |
| `config.yaml` | Training configuration (task type, student/teacher models) |
| `train.jsonl` | ~75 high-quality labeled examples |
| `test.jsonl` | ~78 held-out evaluation examples |
| `unstructured.jsonl` | 1,107 production traces for synthetic data generation |

Upload the data and kick off training with the Distil CLI:

```bash
distil model create my-iot-model
distil model upload-data <model-id> --data ./finetuning-data
distil model run-teacher-evaluation <model-id>
distil model run-training <model-id>
```

### Results

Training a **Qwen3-0.6B** student distilled from an **openai.gpt-oss-120b** teacher, using just ~75 labeled examples and 1,107 unstructured traces:

| Model | ROUGE-L | Tool Call Equivalence |
|-------|---------|----------------------|
| Teacher (openai.gpt-oss-120b) | 87.94% | 50.60% |
| Student base (Qwen3-0.6B) | 50.02% | 9.64% |
| **Student tuned (Qwen3-0.6B)** | **96.89%** | **78.31%** |

The tuned 0.6B model outperforms the 120B teacher on both metrics. This happens because the student learns to match the exact output format from the training data, while the teacher sometimes produces verbose or slightly off-format responses. A model 200x smaller, running faster and cheaper, doing the job better.

## Stage 3: Deploy the Trained Model

With a trained model in hand, the final step is making it available for inference. You can publish it to Hugging Face for others to use, deploy it to managed infrastructure for production traffic, or run it locally for development.

### Upload to Hugging Face

Push the trained model to a Hugging Face repository using the [Distil Labs API](https://docs.distillabs.ai/api-reference/trainings/upload-huggingface-model-trainings-training-id-huggingface-models-post). This requires a Distil Labs API token (see [Appendix: Getting a Distil Labs API Token](#appendix-getting-a-distil-labs-api-token)).

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

This provisions a vLLM-based endpoint and returns the URL and API key. Deployment takes a few minutes to build.

### Test the Deployment

Once deployed, get the client script to test the model:

```bash
distil model invoke <model-id>
```

This outputs the runtime commend for a Python script that queries your deployed model via the OpenAI-compatible API.

Example responses from the deployed model:

```
"turn on the kitchen lights"     → iot_hue_lighton(house_place="kitchen")
"make me a coffee at 7am"        → iot_coffee(time="7am")
"dim the bedroom lights"         → iot_hue_lightdim(house_place="bedroom")
"turn off the smart plug"        → iot_wemo_off(device_type="smart plug")
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

## Project Structure

```
├── README.md                          # This file
├── stage1-preprocess-data.py          # Extract & upload production traces
├── stage2-prepare-distil-labs-data.py # Curate training data & prepare upload
├── finetuning-data/
│   ├── job_description.json           # Task + tool schemas
│   ├── config.yaml                    # Training configuration
│   ├── train.jsonl                    # Labeled training examples
│   ├── test.jsonl                     # Held-out evaluation set
│   └── unstructured.jsonl             # Full production traces
├── benchmark.md                       # Training run details and results
└── data/                              # Raw MASSIVE dataset
```

## Appendix: Getting a Distil Labs API Token

The Distil Labs API requires a bearer token for authentication. Tokens are valid for 1 hour. See the [authentication docs](https://docs.distillabs.ai/getting-started/authentication#api-authentication) for details.

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
