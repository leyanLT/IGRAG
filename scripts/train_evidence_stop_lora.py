import argparse
import json
import os
import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


DEFAULT_DATA_PATH = Path("/root/autodl-tmp/FlashRAG_Data/train_data/hotpotqa_all.jsonl")
DEFAULT_MODEL_PATH = Path("/root/autodl-tmp/models/Qwen/Qwen2.5-7B-Instruct")
DEFAULT_OUTPUT_DIR = Path("/root/autodl-tmp/models/Qwen/Qwen2.5-7B-Instruct-evidence-stop-lora")
IGNORE_INDEX = -100
TRUNCATION_MARKER = "\n...[PROMPT_TRUNCATED]...\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Qwen2.5 evidence stopper with LoRA.")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite-output-dir", action="store_true")

    parser.add_argument("--max-seq-length", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--valid-ratio", type=float, default=0.1)

    parser.add_argument("--num-train-epochs", type=float, default=2)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--eval-steps", type=int, default=500)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--logging-steps", type=int, default=10)

    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)

    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-valid-samples", type=int, default=None)
    parser.add_argument(
        "--tokenization-stats-sample-size",
        type=int,
        default=1000,
        help="Samples per split for pre-training tokenization stats. Use 0 for full stats.",
    )

    args = parser.parse_args()
    validate_args(args, parser)
    return args


def validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if not args.data_path.exists():
        parser.error(f"--data-path does not exist: {args.data_path}")
    if not args.model_path.exists():
        parser.error(f"--model-path does not exist: {args.model_path}")
    if args.max_seq_length <= 0:
        parser.error("--max-seq-length must be greater than 0.")
    if not 0 < args.train_ratio < 1:
        parser.error("--train-ratio must be in (0, 1).")
    if not 0 <= args.valid_ratio < 1:
        parser.error("--valid-ratio must be in [0, 1).")
    if args.train_ratio + args.valid_ratio >= 1:
        parser.error("--train-ratio + --valid-ratio must be less than 1.")
    if args.max_train_samples is not None and args.max_train_samples <= 0:
        parser.error("--max-train-samples must be greater than 0.")
    if args.max_valid_samples is not None and args.max_valid_samples <= 0:
        parser.error("--max-valid-samples must be greater than 0.")
    if args.tokenization_stats_sample_size < 0:
        parser.error("--tokenization-stats-sample-size must be greater than or equal to 0.")


def get_rank() -> int:
    return int(os.environ.get("RANK", "0"))


def is_main_process() -> bool:
    return get_rank() == 0


def distributed_barrier() -> None:
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            dist.barrier()
    except Exception:
        return


def prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    if not is_main_process():
        return

    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"Output directory is not empty: {output_dir}. "
                "Pass --overwrite-output-dir to replace it."
            )
        if output_dir.resolve() == Path("/"):
            raise ValueError("Refusing to remove root directory.")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    samples = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
            validate_raw_sample(item, path, line_no)
            samples.append(item)
    if not samples:
        raise ValueError(f"No samples loaded from {path}")
    return samples


def validate_raw_sample(item: Dict[str, Any], path: Path, line_no: int) -> None:
    for key in ("messages", "task", "dataset_name", "split", "item_id"):
        if key not in item:
            raise ValueError(f"Missing `{key}` at {path}:{line_no}")
    messages = item["messages"]
    if not isinstance(messages, list) or len(messages) != 3:
        raise ValueError(f"`messages` must contain exactly 3 messages at {path}:{line_no}")
    expected_roles = ["system", "user", "assistant"]
    for message, role in zip(messages, expected_roles):
        if message.get("role") != role or not isinstance(message.get("content"), str):
            raise ValueError(f"Invalid `{role}` message at {path}:{line_no}")


def group_key(item: Dict[str, Any]) -> Tuple[str, str, str]:
    return (
        str(item["dataset_name"]),
        str(item["split"]),
        str(item["item_id"]),
    )


def split_by_item_id(
    samples: List[Dict[str, Any]],
    train_ratio: float,
    valid_ratio: float,
    seed: int,
) -> Dict[str, List[Dict[str, Any]]]:
    groups = defaultdict(list)
    for sample in samples:
        groups[group_key(sample)].append(sample)

    keys = list(groups.keys())
    random.Random(seed).shuffle(keys)

    train_count = int(len(keys) * train_ratio)
    valid_count = int(len(keys) * valid_ratio)
    train_keys = set(keys[:train_count])
    valid_keys = set(keys[train_count : train_count + valid_count])
    test_keys = set(keys[train_count + valid_count :])

    splits = {"train": [], "valid": [], "test": []}
    for key in keys:
        if key in train_keys:
            split_name = "train"
        elif key in valid_keys:
            split_name = "valid"
        else:
            split_name = "test"
        splits[split_name].extend(groups[key])

    return splits


def save_jsonl(samples: Iterable[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")


def summarize_samples(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    task_counts = Counter()
    should_stop_counts = Counter()
    assistant_json_valid = 0
    select_candidate_invalid = 0
    stop_candidate_invalid = 0

    for sample in samples:
        task = sample.get("task")
        task_counts[task] += 1
        should_stop = sample.get("metadata", {}).get("should_stop")
        should_stop_counts[str(should_stop)] += 1

        try:
            json.loads(sample["messages"][2]["content"])
            assistant_json_valid += 1
        except json.JSONDecodeError:
            pass

        try:
            payload = json.loads(sample["messages"][1]["content"])
        except json.JSONDecodeError:
            payload = {}
        candidate_ids = {
            candidate.get("candidate_id")
            for candidate in payload.get("remaining_candidates", [])
        }
        label = sample.get("label", {})
        if task == "select_candidate" and label.get("selected_candidate_id") not in candidate_ids:
            select_candidate_invalid += 1
        if task == "judge_stop":
            best_remaining_candidate_id = label.get("best_remaining_candidate_id")
            if best_remaining_candidate_id is not None and best_remaining_candidate_id not in candidate_ids:
                stop_candidate_invalid += 1

    group_count = len({group_key(sample) for sample in samples})
    return {
        "line_count": len(samples),
        "group_count": group_count,
        "avg_lines_per_group": len(samples) / group_count if group_count else 0,
        "task_counts": dict(task_counts),
        "should_stop_counts": dict(should_stop_counts),
        "assistant_json_valid": assistant_json_valid,
        "assistant_json_invalid": len(samples) - assistant_json_valid,
        "select_candidate_invalid": select_candidate_invalid,
        "stop_candidate_invalid": stop_candidate_invalid,
    }


def save_split_files_and_stats(
    splits: Dict[str, List[Dict[str, Any]]],
    output_dir: Path,
    tokenization_stats: Dict[str, Any],
) -> None:
    data_split_dir = output_dir / "data_splits"
    for split_name, split_samples in splits.items():
        save_jsonl(split_samples, data_split_dir / f"{split_name}.jsonl")

    stats = {
        "raw": {split_name: summarize_samples(split_samples) for split_name, split_samples in splits.items()},
        "tokenization": tokenization_stats,
    }
    with (data_split_dir / "stats.json").open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)


def maybe_limit_samples(samples: List[Dict[str, Any]], limit: int = None) -> List[Dict[str, Any]]:
    if limit is None:
        return samples
    return samples[:limit]


def to_list_ids(ids: Any) -> List[int]:
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return list(ids)


def build_tokenized_example(
    sample: Dict[str, Any],
    tokenizer,
    max_seq_length: int,
) -> Tuple[Dict[str, List[int]], Dict[str, Any]]:
    messages = sample["messages"]
    prompt_messages = messages[:2]
    prompt_ids = to_list_ids(
        tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=True,
            add_generation_prompt=True,
        )
    )
    full_ids = to_list_ids(
        tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
        )
    )

    if full_ids[: len(prompt_ids)] == prompt_ids:
        assistant_ids = full_ids[len(prompt_ids) :]
    else:
        assistant_ids = tokenizer(
            messages[2]["content"] + (tokenizer.eos_token or ""),
            add_special_tokens=False,
        )["input_ids"]

    if len(assistant_ids) > max_seq_length:
        raise ValueError(
            f"Assistant label is longer than max_seq_length for sample {sample.get('sample_id')}: "
            f"{len(assistant_ids)} > {max_seq_length}"
        )

    max_prompt_length = max_seq_length - len(assistant_ids)
    prompt_ids, truncation_stats = truncate_prompt_ids(prompt_ids, tokenizer, max_prompt_length)

    input_ids = prompt_ids + assistant_ids
    labels = [IGNORE_INDEX] * len(prompt_ids) + assistant_ids
    attention_mask = [1] * len(input_ids)

    return (
        {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        },
        {
            **truncation_stats,
            "assistant_tokens": len(assistant_ids),
            "total_tokens": len(input_ids),
        },
    )


def truncate_prompt_ids(
    prompt_ids: List[int],
    tokenizer,
    max_prompt_length: int,
) -> Tuple[List[int], Dict[str, Any]]:
    original_prompt_tokens = len(prompt_ids)
    if original_prompt_tokens <= max_prompt_length:
        return (
            prompt_ids,
            {
                "truncated": False,
                "prompt_tokens": original_prompt_tokens,
                "prompt_tokens_kept": original_prompt_tokens,
                "prompt_head_tokens_kept": original_prompt_tokens,
                "prompt_tail_tokens_kept": 0,
                "truncation_marker_tokens": 0,
            },
        )

    if max_prompt_length <= 0:
        return (
            [],
            {
                "truncated": True,
                "prompt_tokens": original_prompt_tokens,
                "prompt_tokens_kept": 0,
                "prompt_head_tokens_kept": 0,
                "prompt_tail_tokens_kept": 0,
                "truncation_marker_tokens": 0,
            },
        )

    marker_ids = tokenizer(TRUNCATION_MARKER, add_special_tokens=False)["input_ids"]
    if len(marker_ids) >= max_prompt_length:
        kept_prompt_ids = marker_ids[:max_prompt_length]
        return (
            kept_prompt_ids,
            {
                "truncated": True,
                "prompt_tokens": original_prompt_tokens,
                "prompt_tokens_kept": len(kept_prompt_ids),
                "prompt_head_tokens_kept": 0,
                "prompt_tail_tokens_kept": 0,
                "truncation_marker_tokens": len(kept_prompt_ids),
            },
        )

    available = max_prompt_length - len(marker_ids)
    head_len = min(1024, max(1, available // 3))
    tail_len = available - head_len
    if tail_len <= 0:
        head_len = available
        tail_len = 0

    prompt_head = prompt_ids[:head_len]
    prompt_tail = prompt_ids[-tail_len:] if tail_len > 0 else []
    kept_prompt_ids = prompt_head + marker_ids + prompt_tail

    return (
        kept_prompt_ids,
        {
            "truncated": True,
            "prompt_tokens": original_prompt_tokens,
            "prompt_tokens_kept": len(kept_prompt_ids),
            "prompt_head_tokens_kept": len(prompt_head),
            "prompt_tail_tokens_kept": len(prompt_tail),
            "truncation_marker_tokens": len(marker_ids),
        },
    )


def compute_tokenization_stats(
    split_samples: Dict[str, List[Dict[str, Any]]],
    tokenizer,
    max_seq_length: int,
    sample_size: int = 1000,
    seed: int = 2024,
) -> Dict[str, Any]:
    stats = {}
    for split_name, samples in split_samples.items():
        if sample_size and len(samples) > sample_size:
            sampled_samples = random.Random(f"{seed}:{split_name}").sample(samples, sample_size)
        else:
            sampled_samples = samples

        truncated_count = 0
        max_total_tokens = 0
        max_prompt_tokens = 0
        max_assistant_tokens = 0
        min_truncated_prompt_head_tokens_kept = None
        min_truncated_prompt_tail_tokens_kept = None
        for sample in sampled_samples:
            _, item_stats = build_tokenized_example(sample, tokenizer, max_seq_length)
            truncated_count += int(item_stats["truncated"])
            max_total_tokens = max(max_total_tokens, item_stats["total_tokens"])
            max_prompt_tokens = max(max_prompt_tokens, item_stats["prompt_tokens"])
            max_assistant_tokens = max(max_assistant_tokens, item_stats["assistant_tokens"])
            if item_stats["truncated"]:
                min_truncated_prompt_head_tokens_kept = min(
                    min_truncated_prompt_head_tokens_kept
                    if min_truncated_prompt_head_tokens_kept is not None
                    else item_stats["prompt_head_tokens_kept"],
                    item_stats["prompt_head_tokens_kept"],
                )
                min_truncated_prompt_tail_tokens_kept = min(
                    min_truncated_prompt_tail_tokens_kept
                    if min_truncated_prompt_tail_tokens_kept is not None
                    else item_stats["prompt_tail_tokens_kept"],
                    item_stats["prompt_tail_tokens_kept"],
                )
        stats[split_name] = {
            "sample_count": len(samples),
            "stats_sample_count": len(sampled_samples),
            "stats_sampled": len(sampled_samples) != len(samples),
            "truncated_count": truncated_count,
            "truncated_ratio": truncated_count / len(sampled_samples) if sampled_samples else 0,
            "max_total_tokens": max_total_tokens,
            "max_prompt_tokens": max_prompt_tokens,
            "max_assistant_tokens": max_assistant_tokens,
            "min_truncated_prompt_head_tokens_kept": min_truncated_prompt_head_tokens_kept or 0,
            "min_truncated_prompt_tail_tokens_kept": min_truncated_prompt_tail_tokens_kept or 0,
        }
    return stats


class EvidenceStopDataset:
    def __init__(self, samples: List[Dict[str, Any]], tokenizer, max_seq_length: int):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, List[int]]:
        tokenized, _ = build_tokenized_example(
            self.samples[index],
            self.tokenizer,
            self.max_seq_length,
        )
        return tokenized


class CausalLMCollator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, features: List[Dict[str, List[int]]]) -> Dict[str, Any]:
        import torch

        max_len = max(len(feature["input_ids"]) for feature in features)
        input_ids = []
        attention_mask = []
        labels = []
        for feature in features:
            pad_len = max_len - len(feature["input_ids"])
            input_ids.append(feature["input_ids"] + [self.pad_token_id] * pad_len)
            attention_mask.append(feature["attention_mask"] + [0] * pad_len)
            labels.append(feature["labels"] + [IGNORE_INDEX] * pad_len)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def build_training_arguments(args: argparse.Namespace):
    from transformers import TrainingArguments

    kwargs = {
        "output_dir": str(args.output_dir),
        "overwrite_output_dir": True,
        "do_train": True,
        "do_eval": True,
        "num_train_epochs": args.num_train_epochs,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "warmup_ratio": args.warmup_ratio,
        "bf16": True,
        "fp16": False,
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "eval_steps": args.eval_steps,
        "save_total_limit": 2,
        "save_strategy": "steps",
        "evaluation_strategy": "steps",
        "gradient_checkpointing": True,
        "remove_unused_columns": False,
        "report_to": [],
        "optim": "adamw_torch",
        "ddp_find_unused_parameters": False,
    }
    try:
        return TrainingArguments(**kwargs)
    except TypeError as exc:
        if "evaluation_strategy" not in str(exc):
            raise
        kwargs["eval_strategy"] = kwargs.pop("evaluation_strategy")
        return TrainingArguments(**kwargs)


def load_tokenizer(args: argparse.Namespace):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def load_lora_model(args: argparse.Namespace):
    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    model = get_peft_model(model, lora_config)
    if is_main_process():
        model.print_trainable_parameters()
    return model


def main() -> None:
    args = parse_args()
    prepare_output_dir(args.output_dir, args.overwrite_output_dir)
    distributed_barrier()

    from transformers import Trainer, set_seed

    set_seed(args.seed)

    samples = load_jsonl(args.data_path)
    splits = split_by_item_id(samples, args.train_ratio, args.valid_ratio, args.seed)
    train_samples = maybe_limit_samples(splits["train"], args.max_train_samples)
    valid_samples = maybe_limit_samples(splits["valid"], args.max_valid_samples)
    if not train_samples:
        raise ValueError("No training samples after split.")
    if not valid_samples:
        raise ValueError("No validation samples after split.")

    tokenizer = load_tokenizer(args)
    used_splits = {"train": train_samples, "valid": valid_samples}
    if is_main_process():
        tokenization_stats = compute_tokenization_stats(
            used_splits,
            tokenizer,
            args.max_seq_length,
            sample_size=args.tokenization_stats_sample_size,
            seed=args.seed,
        )
        save_split_files_and_stats(splits, args.output_dir, tokenization_stats)
    distributed_barrier()

    model = load_lora_model(args)
    train_dataset = EvidenceStopDataset(train_samples, tokenizer, args.max_seq_length)
    valid_dataset = EvidenceStopDataset(valid_samples, tokenizer, args.max_seq_length)
    collator = CausalLMCollator(tokenizer.pad_token_id)
    training_args = build_training_arguments(args)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=collator,
    )
    trainer.train()
    final_eval_metrics = trainer.evaluate(metric_key_prefix="final_eval")
    trainer.log_metrics("final_eval", final_eval_metrics)
    trainer.save_metrics("final_eval", final_eval_metrics)
    trainer.save_model(str(args.output_dir))

    if is_main_process():
        with (args.output_dir / "final_eval_metrics.json").open("w", encoding="utf-8") as f:
            json.dump(final_eval_metrics, f, indent=2, ensure_ascii=False)
        tokenizer.save_pretrained(args.output_dir)
        print(f"Saved LoRA adapter and tokenizer to {args.output_dir}")


if __name__ == "__main__":
    main()
