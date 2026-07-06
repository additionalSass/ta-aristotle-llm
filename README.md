# Validation of the Decompose-Search-Resolve Framework for Logical Reasoning on Open-Weight Large Language Models (with Qwen and DeepSeek)

A capstone reproduction and validating extension of the **Aristotle Decompose–Search–Resolve framework**, comparing direct prompting, few-shot prompting, and structured logical reasoning for open-weight Large Language Models on ProntoQA and ProofWriter.

> **Project type:** University capstone evaluation project
> 
> **Primary skills:** Python, LLM APIs, prompt evaluation, concurrent processing, experimental design, JSON data pipelines
> 
> **Status:** Experiments completed .

## Overview

Large language models can produce fluent answers while still making errors on deductive-logic problems. This project evaluates whether a structured reasoning framework improves logical question answering compared with simpler prompting methods, relying on open-weight LLMs.

I adapted the implementation associated with the ACL 2025 paper:

**“Aristotle: Mastering Logical Reasoning with a Logic-Complete Decompose-Search-Resolve Framework”**

The original Aristotle framework was created by Xu et al. This repository does **not** claim authorship of that framework. My contribution is an experimental adaptation that:

* Runs the pipeline through the DeepInfra OpenAI-compatible API.
* Evaluates Qwen3-14B, Qwen3-32B, and DeepSeek-R1-0528.
* Adds direct zero-shot and few-shot baselines.
* Compares the methods on ProntoQA and ProofWriter.
* Adds parallel inference and protected JSON output handling.
* Provides separate inference and evaluation workflows.

## Practical Questions

This project investigates:

1. Does the Aristotle Decompose–Search–Resolve workflow outperform direct zero-shot prompting for open-weight large language models?
2. Does few-shot prompting provide a meaningful improvement over direct prompting for open-weight large language models?
3. Do larger open-weight models benefit less from structured decomposition than smaller open-weight models?
4. Are performance patterns the same between ProntoQA and ProofWriter?
5. What types of logical errors can be seen?

## Methods

### 1. True Naive Prompting

The model receives the logical problem and is asked to return an answer directly.

This serves as the simplest zero-shot baseline. It does not include demonstrations and does not require the Aristotle reasoning pipeline.

Main scripts:

```text
true_naive_pt.py
evaluate_true_naive.py
```

### 2. Few-Shot Prompting

The model receives several example problems and answers before processing the target problem.

This tests whether in-context demonstrations improve logical reasoning without using the complete Decompose–Search–Resolve framework.

Main scripts:

```text
fewshot_prompting.py
evaluate_fewshot_pt.py
```

> The prompt file named `naive_prompt.txt` is used by the few-shot method despite its historical filename.

### 3. Decompose–Search–Resolve

This method follows the main structure of the Aristotle framework:

1. **Translation and decomposition**
   Natural-language premises are translated into structured logical representations and decomposed into smaller reasoning units.

2. **Negation initialization**
   A second reasoning path is constructed by negating the target conclusion.

3. **Search and resolution**
   The system evaluates both the original and negated paths.

4. **Final aggregation**
   The outputs from the two paths are combined to determine the final prediction.

Main scripts:

```text
translate_decompose.py
negate.py
search_resolve.py
evaluate.py
```

## Experiment Matrix

| Dataset                  | Qwen3-14B | Qwen3-32B | DeepSeek-R1-0528 |
| ------------------------ | --------: | --------: | ---------------: |
| ProntoQA — True Naive    |       Yes |       Yes |              Yes |
| ProntoQA — Few-Shot      |       Yes |       Yes |              Yes |
| ProntoQA — DSR           |       Yes |       Yes |              Yes |
| ProofWriter — True Naive |       Yes |       Yes |    Not evaluated |
| ProofWriter — Few-Shot   |       Yes |       Yes |    Not evaluated |
| ProofWriter — DSR        |       Yes |       Yes |    Not evaluated |

DeepSeek-R1-0528 was not used for the ProofWriter experiments. Results should not be interpreted as a complete three-model comparison on that dataset.

## Models

Inference scripts use the full DeepInfra model identifier. Local processing and evaluation scripts use a shortened identifier for result-file matching.

| Model            | Inference identifier           | Evaluation identifier |
| ---------------- | ------------------------------ | --------------------- |
| Qwen3-14B        | `Qwen/Qwen3-14B`               | `qwen3-14b`           |
| Qwen3-32B        | `Qwen/Qwen3-32B`               | `qwen3-32b`           |
| DeepSeek-R1-0528 | `deepseek-ai/DeepSeek-R1-0528` | `deepseek`            |

## Datasets

### ProntoQA

ProntoQA contains synthetic deductive-reasoning problems expressed in natural language. A model must determine whether a conclusion follows from the supplied facts and rules.

### ProofWriter

ProofWriter evaluates deductive reasoning over facts and rules with varying proof depths.

This project currently uses the `dev` split for both datasets.

## Engineering Contributions

In addition to adapting the model provider and prompts, this project includes:

* Parallel API requests using Python concurrency utilities.
* Configurable batch execution.
* Progress reporting with `tqdm`.
* Atomic JSON-file access using `filelock`.
* Separate result files for positive and negated search paths.
* Dedicated evaluation scripts for each prompting method.
* Normalized model-name conventions for result lookup.
* Automatic creation of the output directory.

These changes focus on experimental execution and reliability rather than proposing a new reasoning algorithm.

## Repository Structure

```text
.
├── data/
│   ├── ProntoQA/
│   │   └── dev.json
│   └── ProofWriter/
│       └── dev.json
│
├── prompts/
│   ├── ProntoQA/
│   │   ├── and_or_decomposer.txt
│   │   ├── naive_prompt.txt
│   │   ├── translation.txt
│   │   ├── true_naive_prompt.txt
│   │   └── ...
│   └── ProofWriter/
│       └── ...
│
├── results/
│
├── true_naive_pt.py
├── fewshot_prompting.py
├── translate_decompose.py
├── negate.py
├── search_resolve.py
│
├── evaluate_true_naive.py
├── evaluate_fewshot_pt.py
├── evaluate.py
│
├── utils.py
├── requirements.txt
└── README.md
```

## Requirements

* Python 3.8 or newer
* A DeepInfra API key
* Internet access during inference
* Sufficient API credit for the selected models and experiment size

The documented external Python dependencies are:

* `tqdm`
* `filelock`

`utils.py` contains the local model-client implementation used by the inference scripts.

## Installation

Clone the repository and enter its directory:

```bash
git clone YOUR_REPOSITORY_URL
cd YOUR_REPOSITORY_NAME
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Linux or macOS:

```bash
source .venv/bin/activate
```

Activate it on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install the dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If a complete `requirements.txt` has not yet been created, install the currently documented packages:

```bash
python -m pip install tqdm filelock
```

## API-Key Setup

Store the API key in an environment variable rather than writing it directly into source code.

Linux or macOS:

```bash
export DEEPINFRA_API_KEY="YOUR_DEEPINFRA_API_KEY"
```

Windows PowerShell:

```powershell
$env:DEEPINFRA_API_KEY="YOUR_DEEPINFRA_API_KEY"
```

Do not commit API keys, `.env` files, or generated credentials to Git.

## Quick Start

The examples below use:

* Dataset: ProntoQA
* Model: Qwen3-14B
* Split: `dev`
* Parallel batch count: `4`

### Run the True Naive Baseline

Inference:

```bash
python true_naive_pt.py \
    --dataset_name "ProntoQA" \
    --model_name "Qwen/Qwen3-14B" \
    --base_url "https://api.deepinfra.com/v1/openai" \
    --api_key "$DEEPINFRA_API_KEY" \
    --batch_num 4
```

Evaluation:

```bash
python evaluate_true_naive.py \
    --dataset_name "ProntoQA" \
    --model_name "qwen3-14b"
```

### Run the Few-Shot Baseline

Inference:

```bash
python fewshot_prompting.py \
    --dataset_name "ProntoQA" \
    --model_name "Qwen/Qwen3-14B" \
    --base_url "https://api.deepinfra.com/v1/openai" \
    --api_key "$DEEPINFRA_API_KEY" \
    --batch_num 4
```

Evaluation:

```bash
python evaluate_fewshot_pt.py \
    --dataset_name "ProntoQA" \
    --model_name "qwen3-14b"
```

### Run Decompose–Search–Resolve

#### Step 1: Translation and Decomposition

```bash
python translate_decompose.py \
    --dataset_name "ProntoQA" \
    --model_name "Qwen/Qwen3-14B" \
    --base_url "https://api.deepinfra.com/v1/openai" \
    --api_key "$DEEPINFRA_API_KEY" \
    --split dev \
    --max_new_tokens 6144 \
    --batch_num 4
```

#### Step 2: Initialize the Negated Path

This is a local processing step and does not require the API endpoint.

```bash
python negate.py \
    --dataset_name "ProntoQA" \
    --model "qwen3-14b"
```

#### Step 3: Search the Non-Negated Path

```bash
python search_resolve.py \
    --dataset_name "ProntoQA" \
    --model_name "Qwen/Qwen3-14B" \
    --base_url "https://api.deepinfra.com/v1/openai" \
    --api_key "$DEEPINFRA_API_KEY" \
    --split dev \
    --negation False \
    --max_new_tokens 4096 \
    --batch_num 4
```

#### Step 4: Search the Negated Path

```bash
python search_resolve.py \
    --dataset_name "ProntoQA" \
    --model_name "Qwen/Qwen3-14B" \
    --base_url "https://api.deepinfra.com/v1/openai" \
    --api_key "$DEEPINFRA_API_KEY" \
    --split dev \
    --negation True \
    --max_new_tokens 4096 \
    --batch_num 4
```

#### Step 5: Aggregate and Evaluate

This is also a local processing step.

```bash
python evaluate.py \
    --dataset_name "ProntoQA" \
    --model "qwen3-14b"
```

## Results

The experiments I did compared three distinct approaches:

* **True naive prompting:** direct zero-shot prediction.
* **Few-shot prompting:** prediction guided by examples included in the prompt.
* **Decompose–Search–Resolve (DSR):** the multi-stage neurosymbolic reasoning pipeline adapted from the Aristotle framework.

### ProntoQA

| Method     | Model            |   Accuracy | Macro Precision | Macro Recall |
| ---------- | ---------------- | ---------: | --------------: | -----------: |
| True naive | Qwen3-14B        |     0.7180 |          0.7340 |       0.7192 |
| Few-shot   | Qwen3-14B        | **0.9760** |      **1.0000** |   **0.9760** |
| DSR        | Qwen3-14B        |     0.8740 |          0.9712 |       0.8742 |
| True naive | Qwen3-32B        |     0.8640 |          0.8774 |       0.8614 |
| Few-shot   | Qwen3-32B        | **0.9840** |      **1.0000** |   **0.9834** |
| DSR        | Qwen3-32B        |     0.5080 |          0.9622 |       0.5086 |
| True naive | DeepSeek-R1-0528 |     0.9280 |      **1.0000** |       0.9280 |
| Few-shot   | DeepSeek-R1-0528 | **1.0000** |      **1.0000** |   **1.0000** |
| DSR        | DeepSeek-R1-0528 |     0.0180 |          0.8076 |       0.0176 |

Few-shot prompting managed to produce the highest ProntoQA accuracy for every evaluated model. It reached 0.9760 with Qwen3-14B, 0.9840 with Qwen3-32B, and 1.0000 with DeepSeek-R1-0528.

DSR improved Qwen3-14B over the true-naive baseline, increasing accuracy from 0.7180 to 0.8740. However, this improvement did not seem to generalize to the larger models. DSR accuracy fell to 0.5080 with Qwen3-32B and 0.0180 with DeepSeek-R1-0528. These results on the tested open-weight models show that the additional complexity of DSR did not consistently improve performance on ProntoQA. For this comparatively direct true-or-false reasoning task, well-designed few-shot examples were more effective and considerably more reliable.

### ProofWriter

| Method     | Model     |   Accuracy | Macro Precision | Macro Recall |
| ---------- | --------- | ---------: | --------------: | -----------: |
| True naive | Qwen3-14B |     0.5957 |          0.6317 |       0.5759 |
| Few-shot   | Qwen3-14B |     0.7072 |          0.8315 |       0.6769 |
| DSR        | Qwen3-14B | **0.8303** |      **0.8629** |   **0.8344** |
| True naive | Qwen3-32B |     0.6506 |          0.6685 |       0.6517 |
| Few-shot   | Qwen3-32B | **0.8469** |      **0.8797** |       0.8367 |
| DSR        | Qwen3-32B |     0.8419 |          0.8743 |   **0.8479** |

For ProofWriter, DSR was substantially more competitive. With Qwen3-14B, DSR achieved the highest accuracy at 0.8303. This was an improvement of 0.1231 over few-shot prompting and 0.2346 over true-naive prompting. With Qwen3-32B, few-shot prompting achieved the highest accuracy at 0.8469, but DSR was only 0.0050 lower at 0.8419. DSR also obtained the highest macro recall, while few-shot prompting obtained the highest macro precision. DeepSeek-R1-0528 was not evaluated on ProofWriter.

> **Important metric note:** ProofWriter macro precision and macro recall were calculated over the three evaluated ground-truth classes. The `SELF-CONTRADICTORY` output category was excluded from the macro-metric calculation.

## Main Findings

The results suggest that reasoning performance depended on both the dataset and the open-weight language model.

1. **Few-shot prompting was overall the strongest general-purpose method for the open-weight models.**
   It achieved the best accuracy for all models on ProntoQA and remained highly competitive on ProofWriter.

2. **DSR was beneficial for Qwen3-14B on ProofWriter.**
   The structured decomposition and dual-path reasoning process appeared to help the smaller model handle longer or more structured logical inference chains.

3. **DSR did not consistently benefit larger models.**
   On ProntoQA, Qwen3-32B performed substantially worse with DSR than with either baseline. DeepSeek-R1-0528 nearly failed under DSR despite achieving perfect accuracy with few-shot prompting.

4. **A more complex reasoning pipeline was not automatically more effective.**
   DSR required several LLM calls and depended on multiple intermediate representations. Errors introduced during an early stage could propagate through the remainder of the pipeline.

5. **The effectiveness of DSR was task-dependent.**
   It was competitive on ProofWriter but unreliable on ProntoQA. This suggests that structured reasoning frameworks should be selected according to the dataset, model behavior, output-format reliability, and available compute budget.

Overall, the experiments do not support replacing few-shot prompting with DSR in every setting. DSR was most useful for Qwen3-14B on ProofWriter, while few-shot prompting provided the most consistent accuracy across the complete experiment.

## Error Analysis

The DSR failures were grouped into four main categories.

### 1. Normalized-Conjecture Format Errors

The most severe failures occurred when the model did not follow the symbolic format expected by the parser and resolver. The pipeline expected normalized predicates in a format similar to:

```text
Predicate(Entity, BooleanValue)
```

For example:

```text
Small(Alex, False)
Happy(Stella, True)
```

DeepSeek-R1-0528 frequently returned natural-language text, incomplete predicates, or a mixture of natural language and symbolic notation. Examples included outputs equivalent to:

```text
Alex is not small
```

or:

```text
Stella is happy -> Happy(Stella, True)
```

Although these outputs were understandable to a human reader, they did not satisfy the parser's strict schema. Consequently, valid clauses were not added to the search process, and the resolver frequently returned `UNKNOWN`. This interface mismatch was the primary reason for the extremely low DSR accuracy of DeepSeek-R1-0528 on ProntoQA. The result reflects a failure of pipeline compatibility rather than an absence of logical-reasoning capability, as the same model achieved perfect accuracy with few-shot prompting.

### 2. Hallucinated Reasoning Steps

Some DSR outputs introduced predicates or intermediate facts that were not supported by the original premises. These fabricated steps could create false contradictions or lead the search toward an invalid conclusion. In several cases, the model produced contradictions on both the original and negated reasoning paths, even though at least one path should have remained logically consistent.

This indicates that the model was sometimes generating plausible-looking symbolic derivations rather than applying the available rules faithfully.

### 3. Incorrect Resolution

In other cases, the model identified relevant facts and rules but applied the resolution principle incorrectly. Typical failures included:

* Resolving against the wrong predicate.
* Failing to eliminate complementary literals.
* Repeating an existing fact without advancing the proof.
* Producing an intermediate conclusion with an incorrect Boolean value.

These mistakes prevented the system from reaching a contradiction or deriving the target conclusion, even when the required evidence was present in the reasoning context.

### 4. Incomplete Search

The search process sometimes stopped before exploring all relevant inference paths. A large language model could complete several correct early steps but fail to continue through the remaining predicates needed to prove or disprove the conjecture. The resolver would then return `UNKNOWN` because no contradiction had been discovered.

This behavior suggests that DSR's search procedure was not guaranteed to be complete when individual search decisions were delegated to an LLM.

## Efficiency and Practical Implications

DSR required substantially more inference work than either baseline because each example passed through translation, decomposition, positive-path search, negated-path search, and final resolution.

For ProntoQA, this additional computation did not provide a corresponding accuracy benefit. Few-shot prompting was both simpler and more accurate.

For ProofWriter, particularly with Qwen3-14B, the additional DSR computation resulted in a meaningful accuracy improvement. DSR may therefore be useful when:

* The task contains longer or more structured inference chains.
* Reasoning accuracy is more important than token cost or latency.
* The selected model reliably follows the required symbolic format.
* Intermediate outputs are validated before being passed to later stages.

A production implementation could consider implementing and using schema-constrained generation, automatic output validation, retry mechanisms, and deterministic symbolic resolution in order to prevent formatting errors from invalidating the complete reasoning pipeline.

## Reproducibility Notes

LLM API outputs from an online platform like Deep Infra may vary between executions. For each reported experiment, it is good to record:

* Execution date.
* Exact provider model identifier.
* Dataset version and split.
* Prompt version or commit hash.
* Maximum output-token setting.
* Batch count.
* Temperature and sampling parameters, when configurable.
* Number of retries and failed requests.
* Evaluation-script version or commit hash.

Generated output files should not be silently overwritten. Preserve the configuration associated with each result.

## Limitations

* This is an adaptation and evaluation project, not the original Aristotle research.
* The project does not introduce a new logical-reasoning algorithm.
* The models were accessed through a hosted API on Deep Infra rather than trained locally.
* No model was fine-tuned as part of this project.
* Only the documented dataset splits were evaluated.
* DeepSeek-R1-0528 was not evaluated on ProofWriter.
* Hosted-model behavior may change when providers update their deployments.
* API failures and nondeterministic outputs can affect reproducibility.
* Accuracy alone does not measure complete reasoning faithfulness.
* The current project is an experimental benchmark, not a production application.

## Future Improvements

Planned improvements can include:

* Add latency and monetary cost results.
* Add automated unit tests for parsers and evaluators.
* Add retry, timeout, and structured logging support.
* Move experiment settings into configuration files.
* Add a single command for running a complete experiment.
* Add GitHub Actions for basic testing.
* Add a Dockerfile for environment reproducibility.
* Add an interactive comparison interface or REST API.
* Evaluate additional datasets and model providers.
* Measure statistical uncertainty across repeated runs.

## Project Provenance and Attribution

This project is based on the original architecture and source code associated with:

* [Aristotle: Mastering Logical Reasoning with a Logic-Complete Decompose-Search-Resolve Framework](https://aclanthology.org/2025.acl-long.153/)
* [Aristotle arXiv paper](https://arxiv.org/abs/2412.16953)
* [Original Aristotle repository](https://github.com/Aiden0526/Aristotle)

The Aristotle framework, original paper, original implementation, datasets, and original prompt designs should be attributed to their respective authors. My work focuses on adapting the execution environment, adding my own ideas of comparative prompting baselines, testing additional and open-weight models, organizing evaluation workflows, and documenting the resulting experiment.


## Citation

When using the original Aristotle framework or discussing its research contribution, cite the original paper:

```bibtex
@inproceedings{xu-etal-2025-aristotle,
    title = "Aristotle: Mastering Logical Reasoning with A Logic-Complete Decompose-Search-Resolve Framework",
    author = "Xu, Jundong  and
      Fei, Hao  and
      Luo, Meng  and
      Liu, Qian  and
      Pan, Liangming  and
      Wang, William Yang  and
      Nakov, Preslav  and
      Lee, Mong-Li  and
      Hsu, Wynne",
    editor = "Che, Wanxiang  and
      Nabende, Joyce  and
      Shutova, Ekaterina  and
      Pilehvar, Mohammad Taher",
    booktitle = "Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2025",
    address = "Vienna, Austria",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.acl-long.153/",
    doi = "10.18653/v1/2025.acl-long.153",
    pages = "3052--3075",
    ISBN = "979-8-89176-251-0",
    abstract = "In the context of large language models (LLMs), current advanced reasoning methods have made impressive strides in various reasoning tasks. However, when it comes to logical reasoning tasks, significant challenges remain in both efficacy and efficiency. This is rooted in the fact that these systems fail to fully leverage the inherent structure of logical tasks throughout the reasoning processes, including decomposition, search, and resolution. To address this, this paper proposes a logic-complete reasoning framework, Aristotle. The framework consists of three key components: Logical Decomposer, Logical Search Router, and Logical Resolver, in which symbolic expressions and logical rules are comprehensively integrated into the entire reasoning process, significantly alleviating the bottlenecks of logical reasoning, i.e., reducing sub-task complexity, minimizing search errors, and resolving logical contradictions. Experimental results demonstrate that Aristotle consistently outperforms state-of-the-art reasoning frameworks in both accuracy and efficiency, particularly excelling in complex logical reasoning scenarios."
}
```

## Author

**VANDER GERALD SUKANDI**
Computer Science graduate currently based in Jakarta, Indonesia

* GitHub: `github.com/additionalSass`
* Email: `vndrgrld2001@gmail.com`
