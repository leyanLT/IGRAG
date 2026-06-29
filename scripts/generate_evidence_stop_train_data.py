import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


OUTPUT_DIR = Path("/root/autodl-tmp/FlashRAG_Data/train_data")


def build_training_selector_class(evidence_selection_stopper_cls):
    class TrainingDataEvidenceSelectionStopper(evidence_selection_stopper_cls):
        """Collect SFT messages while preserving EvidenceSelectionStopper behavior."""

        def __init__(self, config, retriever):
            super().__init__(config, retriever)
            self._current_records: List[Dict[str, Any]] = []

        def select_with_records(
            self,
            query: str,
            evidence_pool: List[Dict[str, Any]],
        ) -> Tuple[List[Dict[str, Any]], Dict[str, Any], List[Dict[str, Any]]]:
            self._current_records = []
            selected_docs, trace = self.select(query, evidence_pool)
            records = self._current_records
            self._current_records = []
            return selected_docs, trace, records

        def _select_candidate(
            self,
            query: str,
            selected_payload: List[Dict[str, Any]],
            candidates: List[Dict[str, Any]],
        ) -> Tuple[int, Dict[str, Any]]:
            payload = {
                "original_query": query,
                "selected_evidence": selected_payload,
                "remaining_candidates": candidates,
            }
            selected_idx, result = super()._select_candidate(query, selected_payload, candidates)
            self._current_records.append(
                {
                    "task": "select_candidate",
                    "system_prompt": self.SELECTION_SYSTEM_PROMPT,
                    "payload": payload,
                    "label": copy.deepcopy(result),
                    "selected_pool_index": selected_idx,
                }
            )
            return selected_idx, result

        def _judge_stop(
            self,
            query: str,
            selected_payload: List[Dict[str, Any]],
            remaining_candidates: List[Dict[str, Any]],
        ) -> Dict[str, Any]:
            payload = {
                "original_query": query,
                "selected_evidence": selected_payload,
                "remaining_candidates": remaining_candidates,
            }
            result = super()._judge_stop(query, selected_payload, remaining_candidates)
            result = self._fallback_invalid_stop_candidate(result, remaining_candidates)
            self._current_records.append(
                {
                    "task": "judge_stop",
                    "system_prompt": self.STOP_SYSTEM_PROMPT,
                    "payload": payload,
                    "label": copy.deepcopy(result),
                }
            )
            return result

        def _fallback_invalid_stop_candidate(
            self,
            result: Dict[str, Any],
            remaining_candidates: List[Dict[str, Any]],
        ) -> Dict[str, Any]:
            best_remaining_candidate_id = result.get("best_remaining_candidate_id")
            candidate_ids = {candidate["candidate_id"] for candidate in remaining_candidates}
            if best_remaining_candidate_id is None or best_remaining_candidate_id in candidate_ids:
                return result

            error = (
                "best_remaining_candidate_id must be null or copied from "
                f"remaining_candidates[].candidate_id, got {best_remaining_candidate_id}"
            )
            return {
                "selected_can_answer_query": False,
                "best_remaining_candidate_id": remaining_candidates[0]["candidate_id"]
                if remaining_candidates
                else None,
                "best_remaining_can_fill_gap": True,
                "key_gap": "",
                "should_stop_precondition": False,
                "reason": f"fallback to continue after invalid candidate id: {error}",
                "error": error,
            }

    return TrainingDataEvidenceSelectionStopper


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate qwen-plus SFT data for evidence selection and stopping."
    )
    parser.add_argument(
        "--output-filename",
        required=True,
        help="Output JSONL filename under /root/autodl-tmp/FlashRAG_Data/train_data.",
    )
    parser.add_argument(
        "--shard-count",
        type=int,
        default=1,
        help="Total number of data shards. Defaults to 1.",
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="Current shard index in [0, shard_count - 1]. Defaults to 0.",
    )
    parser.add_argument(
        "--gpu-id",
        default=None,
        help="Override run_igrag.py gpu_id for this process, e.g. 0.",
    )
    args = parser.parse_args()
    validate_args(args, parser)
    return args


def validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.shard_count <= 0:
        parser.error("--shard-count must be greater than 0.")
    if args.shard_index < 0:
        parser.error("--shard-index must be greater than or equal to 0.")
    if args.shard_index >= args.shard_count:
        parser.error("--shard-index must be less than --shard-count.")


def resolve_output_path(output_filename: str) -> Path:
    output_name = output_filename.strip()
    if not output_name:
        raise ValueError("--output-filename cannot be empty.")

    output_path = Path(output_name)
    if output_path.is_absolute() or output_path.name != output_name or ".." in output_path.parts:
        raise ValueError("--output-filename must be a plain filename, not a path.")

    if output_path.suffix == "":
        output_path = output_path.with_suffix(".jsonl")

    return OUTPUT_DIR / output_path.name


def build_config(gpu_id: Optional[str] = None):
    from flashrag.config import Config
    from run_igrag import config_dict as igrag_config_dict

    config_dict = copy.deepcopy(igrag_config_dict)
    config_dict["disable_save"] = True
    config_dict.setdefault("evidence_selection_stop_setting", {})
    config_dict["evidence_selection_stop_setting"]["failure_policy"] = "fallback"
    if gpu_id is not None:
        config_dict["gpu_id"] = gpu_id
    return Config(config_dict=config_dict)


def build_branch_retrieval_results(
    questions: List[str],
    rewrite_results: List[Dict[str, Any]],
    query_rewriter,
    retriever,
) -> List[List[Dict[str, Any]]]:
    original_queries = [item["original_query"] for item in rewrite_results]
    original_results, original_scores = retriever.batch_search(
        original_queries,
        num=query_rewriter.original_topk,
        return_score=True,
    )

    branch_retrieval_results = []
    rewrite_queries = []
    rewrite_metadata = []

    for item_idx, rewrite_result in enumerate(rewrite_results):
        item_branch_results = [
            {
                "branch_id": "original",
                "branch_query": rewrite_result["original_query"],
                "topk": query_rewriter.original_topk,
                "documents": original_results[item_idx],
                "scores": original_scores[item_idx],
            }
        ]
        branch_retrieval_results.append(item_branch_results)

        for branch in rewrite_result["branches"]:
            if branch["branch_id"] == "original":
                continue
            rewrite_queries.append(branch["query"])
            rewrite_metadata.append((item_idx, branch))

    if rewrite_queries:
        rewrite_results_docs, rewrite_scores = retriever.batch_search(
            rewrite_queries,
            num=query_rewriter.branch_topk,
            return_score=True,
        )
        for (item_idx, branch), documents, scores in zip(
            rewrite_metadata,
            rewrite_results_docs,
            rewrite_scores,
        ):
            branch_retrieval_results[item_idx].append(
                {
                    "branch_id": branch["branch_id"],
                    "branch_query": branch["query"],
                    "topk": query_rewriter.branch_topk,
                    "documents": documents,
                    "scores": scores,
                }
            )

    assert len(branch_retrieval_results) == len(questions)
    return branch_retrieval_results


def build_no_rewrite_branch_results(
    questions: List[str],
    retriever,
) -> List[List[Dict[str, Any]]]:
    retrieval_results, retrieval_scores = retriever.batch_search(questions, return_score=True)
    return [
        [
            {
                "branch_id": "original",
                "branch_query": question,
                "topk": len(documents),
                "documents": documents,
                "scores": scores,
            }
        ]
        for question, documents, scores in zip(questions, retrieval_results, retrieval_scores)
    ]


def build_messages(record: Dict[str, Any]) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": record["system_prompt"]},
        {
            "role": "user",
            "content": json.dumps(record["payload"], ensure_ascii=False),
        },
        {
            "role": "assistant",
            "content": json.dumps(record["label"], ensure_ascii=False),
        },
    ]


def build_sample(
    *,
    config,
    split: str,
    item,
    item_index: int,
    record: Dict[str, Any],
    record_index: int,
) -> Dict[str, Any]:
    round_id = (record_index // 2) + 1
    task_suffix = "select" if record["task"] == "select_candidate" else "stop"
    item_id = item.id if item.id is not None else f"idx_{item_index}"
    sample_id = f"{config['dataset_name']}_{split}_{item_id}_r{round_id}_{task_suffix}"

    metadata = {
        "golden_answers": item.golden_answers,
        "min_selected": config["evidence_selection_stop_setting"]["min_selected"],
        "model": config["evidence_selection_stop_setting"]["model"],
        "stop_reason": None,
        "should_stop": None,
    }
    if record["task"] == "judge_stop":
        label = record["label"]
        selected_count = len(record["payload"]["selected_evidence"])
        selected_can_answer = bool(label.get("selected_can_answer_query", False))
        best_remaining_can_fill_gap = bool(label.get("best_remaining_can_fill_gap", True))
        stop_precondition = selected_can_answer or not best_remaining_can_fill_gap
        should_stop = stop_precondition and selected_count >= metadata["min_selected"]
        metadata["should_stop"] = should_stop
        metadata["stop_reason"] = stop_reason(stop_precondition, should_stop, selected_count)

    return {
        "sample_id": sample_id,
        "task": record["task"],
        "dataset_name": config["dataset_name"],
        "split": split,
        "item_id": item_id,
        "question": item.question,
        "round": round_id,
        "messages": build_messages(record),
        "label": record["label"],
        "metadata": metadata,
    }


def validate_sample(sample: Dict[str, Any]) -> None:
    if len(sample["messages"]) != 3:
        raise ValueError(f"{sample['sample_id']} must contain exactly 3 messages.")
    if sample["task"] == "select_candidate":
        payload = json.loads(sample["messages"][1]["content"])
        candidate_ids = {candidate["candidate_id"] for candidate in payload["remaining_candidates"]}
        selected_candidate_id = sample["label"].get("selected_candidate_id")
        if selected_candidate_id not in candidate_ids:
            raise ValueError(
                f"{sample['sample_id']} selected unknown candidate id: {selected_candidate_id}"
            )
    elif sample["task"] == "judge_stop":
        payload = json.loads(sample["messages"][1]["content"])
        candidate_ids = {candidate["candidate_id"] for candidate in payload["remaining_candidates"]}
        best_remaining_candidate_id = sample["label"].get("best_remaining_candidate_id")
        if best_remaining_candidate_id is not None and best_remaining_candidate_id not in candidate_ids:
            raise ValueError(
                f"{sample['sample_id']} judged unknown best remaining candidate id: "
                f"{best_remaining_candidate_id}"
            )
    else:
        raise ValueError(f"{sample['sample_id']} has unknown task: {sample['task']}")


def stop_reason(stop_precondition: bool, should_stop: bool, selected_count: int) -> str:
    if should_stop:
        return "stop_precondition_met_and_min_selected_reached"
    if stop_precondition:
        return f"stop_precondition_met_but_selected_count_{selected_count}_below_minimum"
    return "continue_missing_evidence_or_remaining_candidate_can_fill_gap"


def generate_samples(config, shard_count: int = 1, shard_index: int = 0) -> List[Dict[str, Any]]:
    from flashrag.pipeline.evidence_pool import EvidencePoolBuilder
    from flashrag.pipeline.evidence_selection_stop import EvidenceSelectionStopper
    from flashrag.pipeline.query_rewrite import QwenTurboQueryRewriter
    from flashrag.utils import get_dataset, get_retriever

    if not config["evidence_selection_stop_setting"].get("enabled", False):
        raise ValueError("evidence_selection_stop_setting.enabled must be true.")

    datasets = get_dataset(config)
    retriever = get_retriever(config)
    query_rewriter: Optional[QwenTurboQueryRewriter] = (
        QwenTurboQueryRewriter(config) if config["use_query_rewrite"] else None
    )
    evidence_pool_builder = EvidencePoolBuilder(config["evidence_pool_setting"])
    training_selector_cls = build_training_selector_class(EvidenceSelectionStopper)
    evidence_selector = training_selector_cls(config, retriever)

    samples = []
    for split in config["split"]:
        dataset = datasets.get(split)
        if dataset is None:
            continue

        shard_items = [
            (item_index, item)
            for item_index, item in enumerate(dataset)
            if item_index % shard_count == shard_index
        ]
        if not shard_items:
            continue

        questions = [item.question for _, item in shard_items]
        if query_rewriter is not None:
            rewrite_results = query_rewriter.batch_rewrite(questions)
            branch_retrieval_results = build_branch_retrieval_results(
                questions,
                rewrite_results,
                query_rewriter,
                retriever,
            )
        else:
            branch_retrieval_results = build_no_rewrite_branch_results(questions, retriever)

        evidence_pools = [
            evidence_pool_builder.build(item_results)
            for item_results in branch_retrieval_results
        ]

        for (item_index, item), evidence_pool in zip(shard_items, evidence_pools):
            _, _, records = evidence_selector.select_with_records(item.question, evidence_pool)
            for record_index, record in enumerate(records):
                sample = build_sample(
                    config=config,
                    split=split,
                    item=item,
                    item_index=item_index,
                    record=record,
                    record_index=record_index,
                )
                validate_sample(sample)
                samples.append(sample)

    return samples


def write_jsonl(samples: List[Dict[str, Any]], output_path: Path) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        raise FileExistsError(f"Output file already exists: {output_path}")

    with output_path.open("w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    output_path = resolve_output_path(args.output_filename)
    if output_path.exists():
        raise FileExistsError(f"Output file already exists: {output_path}")
    config = build_config(gpu_id=args.gpu_id)
    samples = generate_samples(
        config,
        shard_count=args.shard_count,
        shard_index=args.shard_index,
    )
    write_jsonl(samples, output_path)
    print(
        f"Saved {len(samples)} training samples to {output_path} "
        f"(shard {args.shard_index}/{args.shard_count}, gpu_id={config['gpu_id']})"
    )


if __name__ == "__main__":
    main()
