import copy
import hashlib
import json
import os
import re
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from flashrag.config.evidence_selection import apply_dataset_evidence_selection_limits


class EvidenceSelectionStopper:
    """Build a minimal sufficient evidence subset with novelty and a chat model."""

    DEFAULT_SETTING = {
        "enabled": False,
        "model": "qwen-plus",
        "api_key": None,
        "api_key_env": "DASHSCOPE_API_KEY",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "timeout": 60,
        "temperature": 0,
        "max_tokens": 1024,
        "min_selected": 2,
        "max_selected": None,
        "candidate_max_chars": 1200,
        "failure_policy": "raise",
        "response_format": {"type": "json_object"},
    }

    SELECTION_SYSTEM_PROMPT = """You are an evidence selection module for retrieval-augmented question answering.
Select exactly one remaining candidate document that provides the highest marginal gain for answering the original query.

Primary criterion:
- The document content is useful for answering the original query.

Secondary signals:
- dense_score: higher usually means more semantically related to the query.
- novelty: higher means less redundant with already selected evidence.

If content usefulness and secondary signals disagree, prioritize content usefulness. Break ties by the candidate order shown in the input.
selected_candidate_id must be copied exactly from one remaining_candidates[].candidate_id value.
Do not return order, pool_index, document id, title, or selected_evidence[].selected_id.

Return strict JSON only with exactly these keys:
{
  "selected_candidate_id": "candidate id from input",
  "useful_for_query": true,
  "reason": "brief reason"
}
"""

    STOP_SYSTEM_PROMPT = """You are an evidence stopping judge for retrieval-augmented question answering.
Judge whether the selected evidence is already sufficient, and whether the best remaining candidate can fill a critical gap.

Definitions:
- selected_can_answer_query: true only if the selected evidence is sufficient to answer the original query.
- best_remaining_can_fill_gap: true if at least one remaining candidate can add critical missing information.
- should_stop_precondition: true if selected_can_answer_query is true OR best_remaining_can_fill_gap is false.
If best_remaining_candidate_id is not null, it must be copied exactly from one remaining_candidates[].candidate_id value.

Return strict JSON only with exactly these keys:
{
  "selected_can_answer_query": false,
  "best_remaining_candidate_id": null,
  "best_remaining_can_fill_gap": true,
  "key_gap": "brief description of the missing information, or empty string",
  "should_stop_precondition": false,
  "reason": "brief reason"
}
"""

    def __init__(self, config, retriever):
        setting = dict(self.DEFAULT_SETTING)
        setting.update(config["evidence_selection_stop_setting"] or {})
        apply_dataset_evidence_selection_limits(setting, config["dataset_name"])
        self.setting = setting
        self.enabled = bool(setting.get("enabled", False))
        self.retriever = retriever
        self.model = setting["model"]
        self.temperature = setting["temperature"]
        self.max_tokens = int(setting["max_tokens"])
        self.min_selected = int(setting["min_selected"])
        self.max_selected = self._optional_positive_int(setting.get("max_selected"), "max_selected")
        if self.max_selected is not None and self.max_selected < self.min_selected:
            raise ValueError("max_selected must be greater than or equal to min_selected.")
        self.candidate_max_chars = int(setting["candidate_max_chars"])
        self.failure_policy = setting["failure_policy"]
        self.response_format = setting.get("response_format")
        self.client = None

        if self.enabled:
            self._init_client()

    def _init_client(self):
        api_key = self.setting.get("api_key") or os.getenv(self.setting.get("api_key_env", "DASHSCOPE_API_KEY"))
        if not api_key:
            self._handle_failure(ValueError(f"Missing {self.model} API key."), fallback=None)
            return

        from openai import OpenAI

        self.client = OpenAI(
            api_key=api_key,
            base_url=self.setting["base_url"],
            timeout=self.setting["timeout"],
        )

    def select(self, query: str, evidence_pool: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        if not self.enabled:
            return evidence_pool, {"enabled": False, "rounds": [], "stop_reason": "disabled"}

        trace = {
            "enabled": True,
            "model": self.model,
            "min_selected": self.min_selected,
            "max_selected": self.max_selected,
            "rounds": [],
            "stop_reason": None,
        }

        if not evidence_pool:
            trace["stop_reason"] = "empty_evidence_pool"
            return [], trace

        docs = [copy.deepcopy(doc) for doc in evidence_pool]
        embeddings = self._encode_documents(docs)
        remaining_indices = list(range(len(docs)))
        selected_indices = []

        while remaining_indices:
            novelty_by_index = self._compute_novelty(embeddings, remaining_indices, selected_indices)
            candidates = self._build_candidate_payload(docs, remaining_indices, novelty_by_index)
            selected_payload = self._build_selected_payload(docs, selected_indices)

            selected_idx, selection_result = self._select_candidate(query, selected_payload, candidates)
            selected_indices.append(selected_idx)
            remaining_indices.remove(selected_idx)

            round_trace = {
                "round": len(trace["rounds"]) + 1,
                "selected_before": [self._doc_ref(docs[idx], idx) for idx in selected_indices[:-1]],
                "candidate_stats": [
                    {
                        "candidate_id": item["candidate_id"],
                        "pool_index": item["pool_index"],
                        "dense_score": item["dense_score"],
                        "novelty": item["novelty"],
                    }
                    for item in candidates
                ],
                "selection": selection_result,
                "selected_doc": self._doc_ref(docs[selected_idx], selected_idx),
            }

            if self.max_selected is not None and len(selected_indices) >= self.max_selected:
                round_trace["stop_judgement"] = None
                round_trace["should_stop"] = True
                round_trace["stop_reason"] = "max_selected_reached"
                trace["rounds"].append(round_trace)
                trace["stop_reason"] = "max_selected_reached"
                break

            if not remaining_indices:
                round_trace["stop_judgement"] = None
                round_trace["should_stop"] = True
                round_trace["stop_reason"] = "candidate_pool_exhausted"
                trace["rounds"].append(round_trace)
                trace["stop_reason"] = "candidate_pool_exhausted"
                break

            updated_novelty = self._compute_novelty(embeddings, remaining_indices, selected_indices)
            remaining_candidates = self._build_candidate_payload(docs, remaining_indices, updated_novelty)
            selected_payload = self._build_selected_payload(docs, selected_indices)
            stop_result = self._judge_stop(query, selected_payload, remaining_candidates)
            selected_can_answer = bool(stop_result.get("selected_can_answer_query", False))
            best_remaining_can_fill_gap = bool(stop_result.get("best_remaining_can_fill_gap", True))
            stop_precondition = selected_can_answer or not best_remaining_can_fill_gap
            should_stop = stop_precondition and len(selected_indices) >= self.min_selected

            round_trace["stop_judgement"] = stop_result
            round_trace["stop_precondition"] = stop_precondition
            round_trace["should_stop"] = should_stop
            round_trace["stop_reason"] = self._stop_reason(stop_precondition, should_stop, len(selected_indices))
            trace["rounds"].append(round_trace)

            if should_stop:
                trace["stop_reason"] = round_trace["stop_reason"]
                break

        selected_docs = [docs[idx] for idx in selected_indices]
        trace["selected_count"] = len(selected_docs)
        trace["selected_doc_refs"] = [self._doc_ref(doc, idx) for doc, idx in zip(selected_docs, selected_indices)]
        if trace["stop_reason"] is None:
            trace["stop_reason"] = "completed"
        return selected_docs, trace

    def _select_candidate(
        self,
        query: str,
        selected_payload: List[Dict[str, Any]],
        candidates: List[Dict[str, Any]],
    ) -> Tuple[int, Dict[str, Any]]:
        try:
            result = self._call_qwen(
                self.SELECTION_SYSTEM_PROMPT,
                {
                    "original_query": query,
                    "selected_evidence": selected_payload,
                    "remaining_candidates": candidates,
                },
            )
            candidate_id = result.get("selected_candidate_id")
            for candidate in candidates:
                if candidate["candidate_id"] == candidate_id:
                    result["selected_candidate_id"] = candidate_id
                    return candidate["pool_index"], result
            raise ValueError(f"{self.model} selected unknown candidate id: {candidate_id}")
        except Exception as exc:
            fallback = {
                "selected_candidate_id": candidates[0]["candidate_id"],
                "useful_for_query": None,
                "reason": f"fallback to first candidate after failure: {exc}",
                "error": str(exc),
            }
            self._handle_failure(exc, fallback=fallback)
            return candidates[0]["pool_index"], fallback

    def _judge_stop(
        self,
        query: str,
        selected_payload: List[Dict[str, Any]],
        remaining_candidates: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        try:
            result = self._call_qwen(
                self.STOP_SYSTEM_PROMPT,
                {
                    "original_query": query,
                    "selected_evidence": selected_payload,
                    "remaining_candidates": remaining_candidates,
                },
            )
            result["selected_can_answer_query"] = self._to_bool(result.get("selected_can_answer_query"), default=False)
            result["best_remaining_can_fill_gap"] = self._to_bool(result.get("best_remaining_can_fill_gap"), default=True)
            result["should_stop_precondition"] = self._to_bool(
                result.get("should_stop_precondition"),
                default=result["selected_can_answer_query"] or not result["best_remaining_can_fill_gap"],
            )
            self._validate_best_remaining_candidate_id(result, remaining_candidates)
            return result
        except Exception as exc:
            fallback = {
                "selected_can_answer_query": False,
                "best_remaining_candidate_id": remaining_candidates[0]["candidate_id"] if remaining_candidates else None,
                "best_remaining_can_fill_gap": True,
                "key_gap": "",
                "should_stop_precondition": False,
                "reason": f"fallback to continue after failure: {exc}",
                "error": str(exc),
            }
            self._handle_failure(exc, fallback=fallback)
            return fallback

    def _call_qwen(self, system_prompt: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        if self.client is None:
            raise ValueError(f"{self.model} client is not initialized.")

        params = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if self.response_format:
            params["response_format"] = self.response_format

        response = self.client.chat.completions.create(**params)
        if not response.choices:
            raise ValueError(f"No choices returned from {self.model}.")
        content = response.choices[0].message.content
        if not content:
            raise ValueError(f"Empty response returned from {self.model}.")
        parsed = self._parse_json(content)
        if not isinstance(parsed, dict):
            raise ValueError(f"{self.model} response JSON must be an object.")
        return parsed

    def _encode_documents(self, docs: List[Dict[str, Any]]) -> np.ndarray:
        encoder = self._get_encoder()
        texts = [self._doc_text(doc) for doc in docs]
        embeddings = encoder.encode(texts, is_query=False)
        embeddings = np.asarray(embeddings, dtype=np.float32)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return embeddings / norms

    def _get_encoder(self):
        if hasattr(self.retriever, "encoder"):
            return self.retriever.encoder

        for attr_name in ("retriever_list", "_retriever_list"):
            retriever_list = getattr(self.retriever, attr_name, None)
            if not retriever_list:
                continue
            bge_retrievers = [
                retriever
                for retriever in retriever_list
                if hasattr(retriever, "encoder") and "bge" in getattr(retriever, "retrieval_method", "").lower()
            ]
            if bge_retrievers:
                return bge_retrievers[0].encoder
            for retriever in retriever_list:
                if hasattr(retriever, "encoder"):
                    return retriever.encoder

        raise ValueError("EvidenceSelectionStopper requires a dense retriever with an encoder.")

    @staticmethod
    def _compute_novelty(
        embeddings: np.ndarray,
        remaining_indices: List[int],
        selected_indices: List[int],
    ) -> Dict[int, float]:
        if not selected_indices:
            return {idx: 1.0 for idx in remaining_indices}

        remaining_emb = embeddings[remaining_indices]
        selected_emb = embeddings[selected_indices]
        similarities = np.matmul(remaining_emb, selected_emb.T)
        similarities = np.clip(similarities, -1.0, 1.0)
        max_similarities = similarities.max(axis=1)
        return {
            idx: float(1.0 - max_similarity)
            for idx, max_similarity in zip(remaining_indices, max_similarities)
        }

    def _build_candidate_payload(
        self,
        docs: List[Dict[str, Any]],
        indices: List[int],
        novelty_by_index: Dict[int, float],
    ) -> List[Dict[str, Any]]:
        payload = []
        for order, idx in enumerate(indices, start=1):
            doc = docs[idx]
            candidate_id = self._round_candidate_id(order)
            payload.append(
                {
                    "candidate_id": candidate_id,
                    "order": order,
                    "pool_index": idx,
                    "doc_ref": self._llm_doc_ref(doc, idx),
                    "dense_score": self._dense_score(doc),
                    "novelty": novelty_by_index[idx],
                    "content": self._truncate(self._doc_text(doc)),
                }
            )
        return payload

    def _build_selected_payload(self, docs: List[Dict[str, Any]], indices: List[int]) -> List[Dict[str, Any]]:
        payload = []
        for order, idx in enumerate(indices, start=1):
            payload.append(
                {
                    "selected_id": self._selected_evidence_id(order),
                    "pool_index": idx,
                    "doc_ref": self._llm_doc_ref(docs[idx], idx),
                    "content": self._truncate(self._doc_text(docs[idx])),
                }
            )
        return payload

    def _handle_failure(self, exc: Exception, fallback: Optional[Any]):
        if self.failure_policy == "raise":
            raise exc
        warnings.warn(f"{self.model} evidence selection failed, using fallback: {exc}", UserWarning)
        return fallback

    def _validate_best_remaining_candidate_id(
        self,
        result: Dict[str, Any],
        remaining_candidates: List[Dict[str, Any]],
    ) -> None:
        best_remaining_candidate_id = result.get("best_remaining_candidate_id")
        if best_remaining_candidate_id is None:
            return

        candidate_ids = {candidate["candidate_id"] for candidate in remaining_candidates}
        if best_remaining_candidate_id not in candidate_ids:
            raise ValueError(
                f"{self.model} returned unknown best_remaining_candidate_id: "
                f"{best_remaining_candidate_id}. Expected one of {sorted(candidate_ids)} or null."
            )

    @staticmethod
    def _parse_json(text: str) -> Dict[str, Any]:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, flags=re.DOTALL)
            if match is None:
                raise
            return json.loads(match.group(0))

    @staticmethod
    def _candidate_id(pool_index: int) -> str:
        return f"doc_{pool_index + 1}"

    @staticmethod
    def _round_candidate_id(order: int) -> str:
        return f"cand_{order}"

    @staticmethod
    def _selected_evidence_id(order: int) -> str:
        return f"selected_{order}"

    def _llm_doc_ref(self, doc: Dict[str, Any], pool_index: int) -> Dict[str, Any]:
        return {
            "pool_index": pool_index,
            "id": doc.get("id"),
            "title": self._doc_title(doc),
        }

    def _doc_ref(self, doc: Dict[str, Any], pool_index: int) -> Dict[str, Any]:
        return {
            "candidate_id": self._candidate_id(pool_index),
            "pool_index": pool_index,
            "id": doc.get("id"),
            "title": self._doc_title(doc),
        }

    @staticmethod
    def _doc_text(doc: Dict[str, Any]) -> str:
        contents = doc.get("contents")
        if contents:
            return str(contents)
        return f"{doc.get('title', '')}\n{doc.get('text', '')}".strip()

    def _doc_title(self, doc: Dict[str, Any]) -> str:
        title = doc.get("title")
        if title:
            return str(title)
        content = self._doc_text(doc)
        return content.split("\n", 1)[0] if content else ""

    @staticmethod
    def _dense_score(doc: Dict[str, Any]) -> Optional[float]:
        for key in ("dense_similarity", "dense_score", "score"):
            score = doc.get(key)
            if score is not None:
                try:
                    return float(score)
                except (TypeError, ValueError):
                    return None
        return None

    def _truncate(self, text: str) -> str:
        text = str(text)
        if len(text) <= self.candidate_max_chars:
            return text
        return text[: self.candidate_max_chars] + "\n[TRUNCATED]"

    @staticmethod
    def _stop_reason(stop_precondition: bool, should_stop: bool, selected_count: int) -> str:
        if should_stop:
            return "stop_precondition_met_and_min_selected_reached"
        if stop_precondition:
            return f"stop_precondition_met_but_selected_count_{selected_count}_below_minimum"
        return "continue_missing_evidence_or_remaining_candidate_can_fill_gap"

    @staticmethod
    def _to_bool(value: Any, default: bool = False) -> bool:
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {"true", "yes", "1"}
        return bool(value)

    @staticmethod
    def _optional_positive_int(value: Any, key: str) -> Optional[int]:
        if value is None:
            return None
        if isinstance(value, bool):
            raise ValueError(f"{key} must be a positive integer or None.")
        try:
            parsed = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{key} must be a positive integer or None.") from exc
        if parsed <= 0:
            raise ValueError(f"{key} must be a positive integer or None.")
        return parsed

    @staticmethod
    def doc_key(doc: Dict[str, Any]) -> str:
        doc_id = doc.get("id")
        if doc_id is not None:
            return f"id::{doc_id}"
        contents = EvidenceSelectionStopper._doc_text(doc)
        digest = hashlib.sha1(contents.encode("utf-8", errors="ignore")).hexdigest()
        return f"contents::{digest}"
