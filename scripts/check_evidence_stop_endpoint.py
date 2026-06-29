import argparse
import importlib.util
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple


DEFAULT_BASE_URL = "http://127.0.0.1:8001/v1"
DEFAULT_MODEL = "qwen2.5-evidence-stop-sft"
ROOT_DIR = Path(__file__).resolve().parents[1]
EVIDENCE_STOP_FILE = ROOT_DIR / "flashrag" / "pipeline" / "evidence_selection_stop.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smoke test the local evidence selection/stop OpenAI-compatible endpoint."
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--max-tokens", type=int, default=1024)
    return parser.parse_args()


def parse_json_object(text: str) -> Dict[str, Any]:
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match is None:
            raise
        parsed = json.loads(match.group(0))

    if not isinstance(parsed, dict):
        raise ValueError("Endpoint response must be a JSON object.")
    return parsed


def load_evidence_stop_prompts() -> Tuple[str, str]:
    spec = importlib.util.spec_from_file_location("evidence_selection_stop", EVIDENCE_STOP_FILE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load evidence stop prompts from {EVIDENCE_STOP_FILE}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    stopper_cls = module.EvidenceSelectionStopper
    return stopper_cls.SELECTION_SYSTEM_PROMPT, stopper_cls.STOP_SYSTEM_PROMPT


def call_endpoint(
    client: Any,
    *,
    model: str,
    system_prompt: str,
    payload: Dict[str, Any],
    temperature: float,
    max_tokens: int,
) -> Dict[str, Any]:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    if not response.choices:
        raise ValueError("Endpoint returned no choices.")
    content = response.choices[0].message.content
    if not content:
        raise ValueError("Endpoint returned empty content.")
    return parse_json_object(content)


def candidate_ids(payload: Dict[str, Any]) -> List[str]:
    return [candidate["candidate_id"] for candidate in payload["remaining_candidates"]]


def validate_select_result(result: Dict[str, Any], payload: Dict[str, Any]) -> None:
    selected_candidate_id = result.get("selected_candidate_id")
    valid_ids = candidate_ids(payload)
    if selected_candidate_id not in valid_ids:
        raise ValueError(
            f"Invalid selected_candidate_id {selected_candidate_id!r}; expected one of {valid_ids}."
        )


def validate_stop_result(result: Dict[str, Any], payload: Dict[str, Any]) -> None:
    best_remaining_candidate_id = result.get("best_remaining_candidate_id")
    valid_ids = candidate_ids(payload)
    if best_remaining_candidate_id is not None and best_remaining_candidate_id not in valid_ids:
        raise ValueError(
            f"Invalid best_remaining_candidate_id {best_remaining_candidate_id!r}; "
            f"expected one of {valid_ids} or None."
        )

    for key in ("selected_can_answer_query", "best_remaining_can_fill_gap", "should_stop_precondition"):
        if key not in result:
            raise ValueError(f"Missing required key in stop result: {key}")


def build_select_payload() -> Dict[str, Any]:
    return {
        "original_query": "Which city hosted the 2012 Summer Olympics?",
        "selected_evidence": [],
        "remaining_candidates": [
            {
                "candidate_id": "cand_1",
                "order": 1,
                "pool_index": 0,
                "doc_ref": {"pool_index": 0, "id": "doc_1", "title": "London"},
                "dense_score": 0.82,
                "novelty": 1.0,
                "content": "London hosted the 2012 Summer Olympics.",
            },
            {
                "candidate_id": "cand_2",
                "order": 2,
                "pool_index": 1,
                "doc_ref": {"pool_index": 1, "id": "doc_2", "title": "Beijing"},
                "dense_score": 0.61,
                "novelty": 1.0,
                "content": "Beijing hosted the 2008 Summer Olympics.",
            },
        ],
    }


def build_stop_payload() -> Dict[str, Any]:
    return {
        "original_query": "Which city hosted the 2012 Summer Olympics?",
        "selected_evidence": [
            {
                "selected_id": "selected_1",
                "pool_index": 0,
                "doc_ref": {"pool_index": 0, "id": "doc_1", "title": "London"},
                "content": "London hosted the 2012 Summer Olympics.",
            }
        ],
        "remaining_candidates": [
            {
                "candidate_id": "cand_1",
                "order": 1,
                "pool_index": 1,
                "doc_ref": {"pool_index": 1, "id": "doc_2", "title": "Olympic Games"},
                "dense_score": 0.55,
                "novelty": 0.4,
                "content": "The Summer Olympic Games are held every four years.",
            }
        ],
    }


def main() -> None:
    args = parse_args()

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise SystemExit("Missing dependency `openai`. Install project requirements before running this smoke test.") from exc

    selection_system_prompt, stop_system_prompt = load_evidence_stop_prompts()
    client = OpenAI(api_key=args.api_key, base_url=args.base_url, timeout=args.timeout)

    select_payload = build_select_payload()
    select_result = call_endpoint(
        client,
        model=args.model,
        system_prompt=selection_system_prompt,
        payload=select_payload,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
    validate_select_result(select_result, select_payload)

    stop_payload = build_stop_payload()
    stop_result = call_endpoint(
        client,
        model=args.model,
        system_prompt=stop_system_prompt,
        payload=stop_payload,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
    validate_stop_result(stop_result, stop_payload)

    print("Evidence stopper endpoint smoke test passed.")
    print(json.dumps({"select": select_result, "stop": stop_result}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
