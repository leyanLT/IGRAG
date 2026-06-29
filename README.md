# IGRAG: Paper Reproduction Code

This repository contains the reproduction code for **IGRAG**.

> Paper title: **IGRAG: Information-Gain-Driven Compact Evidence Construction for Retrieval-Augmented Generation**
>
> Citation: see [Citation](#citation).

Main IGRAG-SFT results reported in the paper:

| Dataset | EM ↑ | F1 ↑ | Accuracy ↑ | Avg. #Docs ↓ |
| --- | ---: | ---: | ---: | ---: |
| NQ | 39.80 | 52.36 | - | 2.80 |
| HotpotQA | 33.10 | 45.78 | - | 3.56 |
| 2WikiMultiHopQA | 20.00 | 30.65 | - | 3.87 |
| MultiHopRAG | - | - | 62.30 | 2.94 |

IGRAG is implemented on top of [FlashRAG](https://github.com/RUC-NLPIR/FlashRAG). Following the paper, retrieval is reformulated as **compact sufficient evidence construction** rather than fixed top-k selection. The reproduction contains two high-level stages:

```text
selector-stopper distillation
  Qwen-Plus teacher trajectories
  -> step-wise selection and stopping SFT data
  -> LoRA fine-tuning of Qwen2.5-7B-Instruct
  -> local OpenAI-compatible selector-stopper service

IGRAG inference
  input query
  -> structured query expansion: original query + up to 3 rewritten branches
  -> dense retrieval: top-6 documents for each query branch
  -> multi-query mixture evidence pool construction
       deduplicate documents and keep branch provenance, hit count, branch rank, and dense score
  -> novelty-aware incremental evidence selection
       select useful and non-redundant evidence by estimated marginal information gain
  -> sufficiency-aware dynamic stopping
       stop when selected evidence is sufficient or remaining candidates cannot fill a critical gap
  -> answer generation with the compact sufficient evidence set
  -> evaluation, information-gain curve, and average-k analysis
```

This repository is packaged as an IGRAG paper reproduction codebase. The `flashrag/` package is retained because IGRAG extends FlashRAG components for dataset loading, retrieval, generation, and evaluation. New IGRAG-specific logic is mainly implemented in the pipeline hooks, query rewriting, evidence pool construction, selector-stopper, training scripts, and analysis scripts.

The main reproduction scripts are:


| Script                                    | Dataset           | Purpose                                                |
| ----------------------------------------- | ----------------- | ------------------------------------------------------ |
| `run_igrag.py`                            | NQ / HotpotQA / 2WikiMultiHopQA | Run IGRAG on FlashRAG-format Wiki QA benchmarks.       |
| `run_igrag-mul.py`                        | MultiHopRAG       | Run IGRAG on the included MultiHopRAG split.           |
| `scripts/train_evidence_stop_lora.py`     | Provided SFT data | Fine-tune the evidence selection/stop model with LoRA. |
| `scripts/check_evidence_stop_endpoint.py` | Local endpoint    | Smoke test the OpenAI-compatible evidence-stop server. |
| `generate_information_gain_curve.py`      | Evaluation output | Generate the information gain curve.                   |
| `generate_average_k_plot.py`              | Evaluation output | Generate the average selected evidence count.          |


## Requirements

Recommended environment:

- Python 3.11
- CUDA-capable GPU for generator inference and LoRA fine-tuning
- `faiss-cpu` or `faiss-gpu` for dense retrieval indexes
- `transformers`, `peft`, and `torch` for LoRA training
- `vllm` for OpenAI-compatible local serving
- An OpenAI-compatible API key for query rewriting, such as DashScope `qwen-turbo`

The default reproduction configuration uses:


| Component                 | Default model in scripts    | Notes                                                                                                 |
| ------------------------- | --------------------------- | ----------------------------------------------------------------------------------------------------- |
| Retriever                 | `BAAI/bge-large-en-v1.5`    | Use `cls` pooling. Build a matching FAISS index.                                                      |
| Generator                 | `Llama-3.1-8B-Instruct`     | Used by `run_igrag.py` and `run_igrag-mul.py`.                                                        |
| Query rewriter            | `qwen-turbo`                | Called through an OpenAI-compatible API.                                                              |
| Evidence selector/stopper | `qwen2.5-evidence-stop-sft` | A LoRA adapter fine-tuned from `Qwen2.5-7B-Instruct` with the provided SFT data, then served locally. |


## Installation

Create and activate the conda environment:

```bash
conda create -n igrag python=3.11 -y
conda activate igrag
```

Upgrade the Python packaging tools:

```bash
python -m pip install -U pip setuptools wheel
```

Install PyTorch and vLLM for CUDA 12.8:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install vllm --extra-index-url https://download.pytorch.org/whl/cu128
```

Install the remaining project dependencies and Faiss:

```bash
pip install -r requirements.txt
pip install faiss-cpu
pip check
```

Run all commands from the repository root. If Python cannot find the local `flashrag` package, set:

```bash
export PYTHONPATH="$PWD:$PYTHONPATH"
```

## Data Preparation

### HotpotQA, 2WikiMultiHopQA, and NQ

For these three benchmarks, use the **preprocessed FlashRAG version** instead of downloading the raw original datasets. IGRAG expects the FlashRAG JSONL schema, so the recommended source is the official FlashRAG dataset repository: [RUC-NLPIR/FlashRAG_datasets](https://huggingface.co/datasets/RUC-NLPIR/FlashRAG_datasets).

FlashRAG provides:

- `nq/train.jsonl`, `nq/dev.jsonl`, `nq/test.jsonl`
- `hotpotqa/train.jsonl`, `hotpotqa/dev.jsonl`
- `2wikimultihopqa/train.jsonl`, `2wikimultihopqa/dev.jsonl`
- `retrieval-corpus/wiki18_100w.zip`

Install the Hugging Face Hub CLI if it is not available:

```bash
pip install -U huggingface_hub
```

Download the QA datasets from FlashRAG:

```bash
mkdir -p /root/autodl-tmp/FlashRAG_Data

huggingface-cli download RUC-NLPIR/FlashRAG_datasets \
  --repo-type dataset \
  --local-dir /root/autodl-tmp/FlashRAG_Data \
  --include "nq/*.jsonl" "hotpotqa/*.jsonl" "2wikimultihopqa/*.jsonl"
```

Download and extract the FlashRAG Wikipedia retrieval corpus:

```bash
huggingface-cli download RUC-NLPIR/FlashRAG_datasets \
  --repo-type dataset \
  --local-dir /root/autodl-tmp/FlashRAG_Data \
  --include "retrieval-corpus/wiki18_100w.zip"

cd /root/autodl-tmp/FlashRAG_Data/retrieval-corpus
unzip wiki18_100w.zip
```

After downloading, the expected layout is:

```text
/root/autodl-tmp/FlashRAG_Data/
  nq/
    train.jsonl
    dev.jsonl
    test.jsonl
  hotpotqa/
    train.jsonl
    dev.jsonl
  2wikimultihopqa/
    train.jsonl
    dev.jsonl
  retrieval-corpus/
    wiki18_100w.jsonl
```

If Hugging Face is slow or unavailable in your environment, still use the same FlashRAG files through a Hugging Face mirror, or manually download them from the [FlashRAG dataset page](https://huggingface.co/datasets/RUC-NLPIR/FlashRAG_datasets). Do not mix these preprocessed files with raw HotpotQA, raw 2WikiMultiHopQA, or raw NQ files unless you convert them to the FlashRAG schema first.

### MultiHopRAG

This repository already includes a FlashRAG-compatible MultiHopRAG split:

```text
dataset/MultiHopRAG/dev.jsonl
dataset/MultiHopRAG/corpus.jsonl
```

The expected QA format is:

```json
{"id": "0", "question": "...", "golden_answers": ["..."], "metadata": {...}}
```

The expected corpus format is:

```json
{"id": "0", "contents": "...", "title": "...", "source": "..."}
```

If you start from the original MultiHopRAG release files, convert them with:

```bash
python scripts/prepare_multihoprag.py \
  --dataset-dir /path/to/original/MultiHopRAG \
  --output-dir dataset/MultiHopRAG \
  --split dev
```

### Path Configuration

After downloading HotpotQA, 2WikiMultiHopQA, NQ, and the Wikipedia corpus, update the paths in `run_igrag.py` and any dataset-specific experiment script:

Important: the paths currently written in `run_igrag.py` and `run_igrag-mul.py` are reference paths from the author's local environment. They are not portable defaults. Before running the scripts, replace `data_dir`, `corpus_path`, `index_path`, and every entry in `model2path` with the actual paths on your machine.

```text
/path/to/FlashRAG_Data/
  nq/
    train.jsonl
    dev.jsonl
    test.jsonl
  hotpotqa/
    train.jsonl
    dev.jsonl
  2wikimultihopqa/
    train.jsonl
    dev.jsonl
  retrieval-corpus/
    wiki18_100w.jsonl
  indexes/
    bge_large_en_wiki18_100w_sharded/
```

The scripts assume FlashRAG-style fields: `question`, `golden_answers`, and a JSONL retrieval corpus with `id` and `contents`.

## Build Retrieval Index

IGRAG uses `BAAI/bge-large-en-v1.5` in the provided scripts. Build the FAISS index with the same retriever and pooling method used during evaluation.

### Wiki18 Index for HotpotQA, 2WikiMultiHopQA, and NQ

HotpotQA, 2WikiMultiHopQA, and NQ use the FlashRAG Wikipedia retrieval corpus `retrieval-corpus/wiki18_100w.jsonl`. Build a BGE Large index for this corpus:

```bash
CUDA_VISIBLE_DEVICES=0 python -m flashrag.retriever.index_builder \
  --retrieval_method bge-large-en-v1.5 \
  --model_path /root/autodl-tmp/models/BAAI/bge-large-en-v1.5 \
  --corpus_path /root/autodl-tmp/FlashRAG_Data/retrieval-corpus/wiki18_100w.jsonl \
  --save_dir /root/autodl-tmp/FlashRAG_Data/indexes/bge_large_en_wiki18_100w \
  --use_fp16 \
  --max_length 512 \
  --batch_size 256 \
  --pooling_method cls \
  --faiss_type Flat \
  --save_embedding
```

After indexing, update the corresponding experiment script:

```python
"corpus_path": "/root/autodl-tmp/FlashRAG_Data/retrieval-corpus/wiki18_100w.jsonl",
"index_path": "/root/autodl-tmp/FlashRAG_Data/indexes/bge_large_en_wiki18_100w/bge-large-en-v1.5_Flat.index",
```

Use this Wiki18 index for HotpotQA, 2WikiMultiHopQA, and NQ experiments. If you build a sharded index, set `index_path` to the sharded index directory instead of the `.index` file.

### MultiHopRAG Index

Example for the included MultiHopRAG corpus:

```bash
CUDA_VISIBLE_DEVICES=0 python -m flashrag.retriever.index_builder \
  --retrieval_method bge-large-en-v1.5 \
  --model_path /path/to/BAAI/bge-large-en-v1.5 \
  --corpus_path dataset/MultiHopRAG/corpus.jsonl \
  --save_dir /path/to/indexes/MultiHopRAG_bge_large \
  --use_fp16 \
  --max_length 512 \
  --batch_size 256 \
  --pooling_method cls \
  --faiss_type Flat \
  --save_embedding
```

Then update the following fields in `run_igrag-mul.py`:

```python
"data_dir": "/path/to/IG-RAG/dataset",
"corpus_path": "/path/to/IG-RAG/dataset/MultiHopRAG/corpus.jsonl",
"index_path": "/path/to/indexes/MultiHopRAG_bge_large/bge-large-en-v1.5_Flat.index",
"model2path": {
    "bge-large-en-v1.5": "/path/to/BAAI/bge-large-en-v1.5",
    "llama-3.1-8b-instruct": "/path/to/Llama-3.1-8B-Instruct",
},
```

Important: do not reuse an index built with another embedding model. The index must be built with `bge-large-en-v1.5` if you keep the default retrieval configuration.

## Train Evidence-Stop Model

The evidence-stop model used by IGRAG is **not trained from scratch**. It is a LoRA fine-tuned model based on `Qwen2.5-7B-Instruct`.

Use the original `Qwen2.5-7B-Instruct` checkpoint as `--model-path`, and save only the LoRA adapter to `--output-dir`. During evaluation, this adapter is served together with the same base model under the OpenAI-compatible model name `qwen2.5-evidence-stop-sft`.

The evidence selection/stop training data is already included:

```text
scripts/train_data/train_data_all.jsonl
scripts/train_data/small_train_data.jsonl
```

Use `train_data_all.jsonl` for the full reproduction. It contains 115,655 SFT instances distilled from Qwen-Plus teacher trajectories on the training splits of HotpotQA, 2WikiMultiHopQA, and MuSiQue:

| Source split | Instances |
| --- | ---: |
| HotpotQA train | 37,382 |
| 2WikiMultiHopQA train | 50,011 |
| MuSiQue train | 28,262 |

The SFT tasks cover both `select_candidate` and `judge_stop` decisions. Use `small_train_data.jsonl` for the small-data ablation experiment; it contains 23,377 instances sampled from the same sources.

Small-data ablation LoRA fine-tuning:

```bash
PYTORCH_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=0,1,2,3,4 torchrun --nproc_per_node 5 scripts/train_evidence_stop_lora.py \
  --data-path scripts/train_data/small_train_data.jsonl \
  --model-path /root/autodl-tmp/models/Qwen/Qwen2.5-7B-Instruct \
  --output-dir /root/autodl-tmp/models/Qwen/Qwen2.5-7B-Instruct-evidence-stop-lora-small \
  --overwrite-output-dir \
  --max-seq-length 3072 \
  --train-ratio 0.85 \
  --valid-ratio 0.1 \
  --num-train-epochs 1.5 \
  --per-device-train-batch-size 1 \
  --per-device-eval-batch-size 1 \
  --gradient-accumulation-steps 8 \
  --learning-rate 8e-5 \
  --warmup-ratio 0.03 \
  --eval-steps 100 \
  --save-steps 100 \
  --logging-steps 10 \
  --lora-r 16 \
  --lora-alpha 32 \
  --lora-dropout 0.05 \
  --tokenization-stats-sample-size 1000
```

Full LoRA fine-tuning uses five GPUs in the original reproduction setting. Replace the paths with your local model, data, and output directories:

```bash
PYTORCH_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=0,1,2,3,4 torchrun --nproc_per_node 5 scripts/train_evidence_stop_lora.py \
  --data-path scripts/train_data/train_data_all.jsonl \
  --model-path /root/autodl-tmp/models/Qwen/Qwen2.5-7B-Instruct \
  --output-dir /root/autodl-tmp/models/Qwen/Qwen2.5-7B-Instruct-evidence-stop-lora-full \
  --overwrite-output-dir \
  --max-seq-length 3072 \
  --train-ratio 0.85 \
  --valid-ratio 0.1 \
  --num-train-epochs 1.5 \
  --per-device-train-batch-size 1 \
  --per-device-eval-batch-size 1 \
  --gradient-accumulation-steps 8 \
  --learning-rate 8e-5 \
  --warmup-ratio 0.03 \
  --eval-steps 500 \
  --save-steps 500 \
  --logging-steps 10 \
  --lora-r 16 \
  --lora-alpha 32 \
  --lora-dropout 0.05 \
  --tokenization-stats-sample-size 1000
```

If you use the training data shipped with this repository, set `--data-path` to `scripts/train_data/train_data_all.jsonl`.

The training script saves:

```text
/path/to/qwen2.5-evidence-stop-lora/
  adapter_config.json
  adapter_model.safetensors
  data_splits/
    train.jsonl
    valid.jsonl
    test.jsonl
    stats.json
```

## Serve Evidence-Stop Model

The IGRAG evaluation scripts call the evidence-stop model through an OpenAI-compatible API endpoint. Serve the `Qwen2.5-7B-Instruct` base model with the trained LoRA adapter, and expose it as `qwen2.5-evidence-stop-sft`.

A common vLLM serving command is:

```bash
CUDA_VISIBLE_DEVICES=1 vllm serve /root/autodl-tmp/models/Qwen/Qwen2.5-7B-Instruct \
  --enable-lora \
  --lora-modules qwen2.5-evidence-stop-sft=/root/autodl-tmp/models/Qwen/Qwen2.5-7B-Instruct-evidence-stop-lora-full \
  --host 127.0.0.1 \
  --port 8001 \
  --dtype bfloat16 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90 \
  --max-lora-rank 64 \
  --trust-remote-code
```

If your vLLM version uses the older module entrypoint, use:

```bash
CUDA_VISIBLE_DEVICES=1 python -m vllm.entrypoints.openai.api_server \
  --model /root/autodl-tmp/models/Qwen/Qwen2.5-7B-Instruct \
  --enable-lora \
  --lora-modules qwen2.5-evidence-stop-sft=/root/autodl-tmp/models/Qwen/Qwen2.5-7B-Instruct-evidence-stop-lora-full \
  --host 127.0.0.1 \
  --port 8001 \
  --dtype bfloat16 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90 \
  --max-lora-rank 64 \
  --trust-remote-code
```

Check the local endpoint:

```bash
python scripts/check_evidence_stop_endpoint.py \
  --base-url http://127.0.0.1:8001/v1 \
  --model qwen2.5-evidence-stop-sft
```

Expected final line:

```text
Evidence stopper endpoint smoke test passed.
```

## Run IGRAG Evaluation

Before running evaluation, check `run_igrag.py` and `run_igrag-mul.py` carefully. The hard-coded paths in these files are examples only and must be changed to your own dataset, corpus, index, generator model, retriever model, and evidence-stop endpoint paths.

### API key for query rewriting

Query rewriting uses `qwen-turbo` through an OpenAI-compatible API. Set the key with an environment variable:

```bash
export DASHSCOPE_API_KEY="your_api_key_here"
```

Security note: this README intentionally contains no real API keys. Do not commit provider keys to the repository. If your local copy has a real key in `query_rewrite_setting.api_key`, remove it or set it to `None` before sharing the code, and use `DASHSCOPE_API_KEY` instead.

### MultiHopRAG

After building the MultiHopRAG index and starting the evidence-stop endpoint:

```bash
python run_igrag-mul.py
```

`run_igrag-mul.py` defaults to:

- dataset: `MultiHopRAG`
- split: `dev`
- sample count: `1000`
- random sampling: enabled
- retriever: `bge-large-en-v1.5`
- retrieval pooling: `cls`
- base/fallback retrieval top-k: `5`
- query rewriting: enabled
- query rewrite model: `qwen-turbo`
- query rewrite branches: up to `3`
- original query top-k: `6`
- rewritten branch top-k: `6`
- multi-query evidence pool: original query plus up to three rewritten branches, with top-6 documents retrieved per branch
- evidence selection/stop: enabled
- evidence-stop model: `qwen2.5-evidence-stop-sft`
- candidate max chars: `800`
- generator: `llama-3.1-8b-instruct`
- generator max input length: `2048`
- generation max tokens: `64`
- information gain metric: `acc`
- information gain curve max-k: `10`
- average-k plot: enabled through `enable_average_k_plot`

For full-dataset reproduction, edit the script:

```python
"test_sample_num": None,
"random_sample": False,
```

### NQ, HotpotQA, and 2WikiMultiHopQA

After preparing the FlashRAG-format NQ, HotpotQA, or 2WikiMultiHopQA data, building the BGE Wiki18 index, and starting the evidence-stop endpoint:

```bash
python run_igrag.py
```

`run_igrag.py` defaults to:

- dataset: `hotpotqa`
- split: `dev`
- sample count: `1000`
- random sampling: enabled
- retriever: `bge-large-en-v1.5`
- retrieval pooling: `cls`
- base/fallback retrieval top-k: `5`
- query rewriting: enabled
- query rewrite model: `qwen-turbo`
- query rewrite branches: up to `3`
- original query top-k: `6`
- rewritten branch top-k: `6`
- multi-query evidence pool: original query plus up to three rewritten branches, with top-6 documents retrieved per branch
- evidence selection/stop: enabled
- evidence-stop model: `qwen2.5-evidence-stop-sft`
- candidate max chars: `800`
- generator: `llama-3.1-8b-instruct`
- generator max input length: `2048`
- generation max tokens: `64`
- information gain metric: `f1`
- information gain curve max-k: `10`
- average-k plot: enabled through `enable_average_k_plot`

For full-dataset reproduction, edit the script:

```python
"test_sample_num": None,
"random_sample": False,
```

## Outputs and Plots

Each run creates a timestamped directory under `output/`:

```text
output/{dataset_name}_{YYYY_MM_DD_HH_MM}_{save_note}/
  config.yaml
  metric_score.txt
  intermediate_data.json
  igrag_information_gain_{f1_or_acc}.csv
  igrag_information_gain_{f1_or_acc}.json
  igrag_information_gain_{f1_or_acc}.png
  igrag_average_k.csv
  igrag_average_k.json
  igrag_average_k.png
```

Main outputs:

- `metric_score.txt`: aggregate EM/F1/ACC and retrieval metrics.
- `intermediate_data.json`: predictions, retrieved documents, evidence pools, selected evidence, and evidence selection traces.
- `igrag_information_gain_*.png`: average cumulative information gain curve.
- `igrag_average_k.csv`: compact summary with `avg_k` and `valid_sample_count`.
- `igrag_average_k.json`: the same average-k summary plus the source field definition.
- `igrag_average_k.png`: single-value plot of the average selected evidence count `k`.

The plotting switch in the experiment scripts is `enable_average_k_plot`, and the generated file stem is `igrag_average_k`.

The generated `config.yaml` is important for reproducibility because it records the resolved model paths, dataset paths, retrieval settings, and generation settings.

## Reproducibility Notes

- Keep the retriever model, pooling method, and index consistent. The default is `bge-large-en-v1.5` with `cls` pooling.
- Use the same generator model revision when comparing with paper results.
- Record GPU type, CUDA version, PyTorch version, vLLM version, and model checkpoints.
- The default scripts sample a subset for faster debugging. Set `test_sample_num` to `None` and `random_sample` to `False` for full evaluation.
- The evidence-stop endpoint must be running before `run_igrag.py` or `run_igrag-mul.py`.
- Query rewriting depends on an external OpenAI-compatible service. API failures fall back only if `failure_policy` is set to `fallback`.
- `scripts/generate_evidence_stop_train_data.py` is provided for regenerating SFT data, but the released reproduction path uses the included files under `scripts/train_data/`.

## Repository Layout

```text
IG-RAG/
  run_igrag.py
  run_igrag-mul.py
  flashrag/
    pipeline/
      pipeline.py
      query_rewrite.py
      evidence_pool.py
      evidence_selection_stop.py
  scripts/
    train_data/
      train_data_all.jsonl
      small_train_data.jsonl
    train_evidence_stop_lora.py
    check_evidence_stop_endpoint.py
    prepare_multihoprag.py
  dataset/
    MultiHopRAG/
      dev.jsonl
      corpus.jsonl
  generate_information_gain_curve.py
  generate_average_k_plot.py
```

## Citation

Please cite the IGRAG paper when using this reproduction code:

```bibtex
@misc{igrag2026,
  title  = {IGRAG: Information-Gain-Driven Compact Evidence Construction for Retrieval-Augmented Generation},
  author = {Zhang, Zhicheng and Yao, Yuchen and Zhou, Wentao and Sun, Jinran and Lu, Haohao and Cao, Lei and Li, Jie},
  year   = {2026},
  note   = {Preprint submitted to Elsevier},
  url    = {https://github.com/leyanLT/IGRAG}
}
```

This repository is built on FlashRAG. Please also cite FlashRAG:

```bibtex
@article{jin2024flashrag,
  title   = {FlashRAG: A Modular Toolkit for Efficient Retrieval-Augmented Generation Research},
  author  = {Jin, Jiajie and Zhu, Yutao and Yang, Xinyu and Zhang, Chenghao and Dou, Zhicheng and Wen, Ji-Rong},
  journal = {CoRR},
  volume  = {abs/2405.13576},
  year    = {2024},
  url     = {https://arxiv.org/abs/2405.13576},
  eprinttype = {arXiv},
  eprint  = {2405.13576}
}
```

## License

See [LICENSE](LICENSE).
