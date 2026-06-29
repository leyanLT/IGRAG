import json
import os
import re
from typing import Any, Dict, List
import torch
import warnings
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from flashrag.retriever.encoder import Encoder


class BaseReranker:
    r"""Base object for all rerankers."""

    def __init__(self, config):
        self.config = config
        self.reranker_model_name = config["rerank_model_name"]
        self.reranker_model_path = config["rerank_model_path"]
        self.topk = config["rerank_topk"]
        self.max_length = config["rerank_max_length"]
        self.batch_size = config["rerank_batch_size"]
        self.device = config["device"]

    def get_rerank_scores(self, query_list: List[str], doc_list: List[str], batch_size):
        """Return flatten list of scores for each (query,doc) pair
        Args:
            query_list: List of N queries
            doc_list:  Nested list of length N, each element corresponds to K documents of a query

        Return:
            [score(q1,d1), score(q1,d2),... score(q2,d1),...]
        """
        all_scores = []
        return all_scores

    @torch.inference_mode(mode=True)
    def rerank(self, query_list, doc_list, batch_size=None, topk=None):
        r"""Rerank doc_list."""
        if batch_size is None:
            batch_size = self.batch_size
        if topk is None:
            topk = self.topk
        if isinstance(query_list, str):
            query_list = [query_list]
        if not isinstance(doc_list[0], list):
            doc_list = [doc_list]

        assert len(query_list) == len(doc_list)
        if topk > min([len(docs) for docs in doc_list]):
            warnings.warn("The number of doc returned by the retriever is less than the topk.")

        # get doc contents
        doc_contents = []
        for docs in doc_list:
            if all([isinstance(doc, str) for doc in docs]):
                doc_contents.append([doc for doc in docs])
            else:
                doc_contents.append([doc["contents"] for doc in docs])

        all_scores = self.get_rerank_scores(query_list, doc_contents, batch_size)
        assert len(all_scores) == sum([len(docs) for docs in doc_list])

        # sort docs
        start_idx = 0
        final_scores = []
        final_docs = []
        for docs in doc_list:
            doc_scores = all_scores[start_idx : start_idx + len(docs)]
            doc_scores = [float(score) for score in doc_scores]
            sort_idxs = np.argsort(doc_scores)[::-1][:topk]
            start_idx += len(docs)

            final_docs.append([docs[idx] for idx in sort_idxs])
            final_scores.append([doc_scores[idx] for idx in sort_idxs])

        return final_docs, final_scores


class CrossReranker(BaseReranker):
    def __init__(self, config):
        super().__init__(config)
        self.tokenizer = AutoTokenizer.from_pretrained(self.reranker_model_path)
        self.ranker = AutoModelForSequenceClassification.from_pretrained(self.reranker_model_path, num_labels=1)
        self.ranker.eval()
        self.ranker.to(self.device)

    @torch.inference_mode(mode=True)
    def get_rerank_scores(self, query_list, doc_list, batch_size):
        # flatten all pairs
        all_pairs = []
        for query, docs in zip(query_list, doc_list):
            all_pairs.extend([[query, doc] for doc in docs])
        all_scores = []
        for start_idx in tqdm(range(0, len(all_pairs), batch_size), desc="Reranking process: "):
            pair_batch = all_pairs[start_idx : start_idx + batch_size]

            inputs = self.tokenizer(
                pair_batch, padding=True, truncation=True, return_tensors="pt", max_length=self.max_length
            ).to(self.device)
            batch_scores = (
                self.ranker(**inputs, return_dict=True)
                .logits.view(
                    -1,
                )
                .float()
                .cpu()
            )
            all_scores.extend(batch_scores)

        return all_scores


class BiReranker(BaseReranker):
    def __init__(self, config):
        super().__init__(config)
        self.encoder = Encoder(
            model_name=self.reranker_model_name,
            model_path=self.reranker_model_path,
            pooling_method=config["rerank_pooling_method"],
            max_length=self.max_length,
            use_fp16=config["rerank_use_fp16"],
        )

    def get_rerank_scores(self, query_list, doc_list, batch_size):
        query_emb = []
        for start_idx in range(0, len(query_list), batch_size):
            query_batch = query_list[start_idx : start_idx + batch_size]
            batch_emb = self.encoder.encode(query_batch, is_query=True)
            query_emb.append(batch_emb)
        query_emb = np.concatenate(query_emb, axis=0)

        flat_doc_list = sum(doc_list, [])
        doc_emb = []
        for start_idx in range(0, len(flat_doc_list), batch_size):
            doc_batch = flat_doc_list[start_idx : start_idx + batch_size]
            batch_emb = self.encoder.encode(doc_batch, is_query=False)
            doc_emb.append(batch_emb)
        doc_emb = np.concatenate(doc_emb, axis=0)

        scores = query_emb @ doc_emb.T  # K*L
        all_scores = []
        score_idx = 0
        for idx, doc in enumerate(doc_list):
            all_scores.extend(scores[idx, score_idx : score_idx + len(doc)])
            score_idx += len(doc)

        return all_scores


class LLMReranker(BaseReranker):
    r"""Listwise reranker backed by an OpenAI-compatible chat API."""

    DEFAULT_SETTING = {
        "model": "qwen-turbo",
        "api_key": None,
        "api_key_env": "DASHSCOPE_API_KEY",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "timeout": 60,
        "temperature": 0,
        "max_tokens": 256,
        "candidate_max_chars": 1200,
        "failure_policy": "raise",
        "response_format": {"type": "json_object"},
    }

    SYSTEM_PROMPT = """You are a listwise reranking module for retrieval-augmented question answering.
Rank the candidate documents by how useful they are for answering the question.
Return strict JSON only with exactly this shape:
{
  "ranked_doc_ids": ["doc_1", "doc_2"]
}
The ranked_doc_ids array must contain candidate doc_id values, ordered from most relevant to least relevant.
Do not include explanations or any text outside the JSON object.
"""

    def __init__(self, config):
        super().__init__(config)
        setting = dict(self.DEFAULT_SETTING)
        llm_rerank_setting = config["llm_rerank_setting"] if "llm_rerank_setting" in config else None
        setting.update(llm_rerank_setting or {})
        self.setting = setting

        self.model = setting.get("model") or self.reranker_model_name
        self.temperature = setting["temperature"]
        self.max_tokens = int(setting["max_tokens"])
        self.candidate_max_chars = int(setting["candidate_max_chars"])
        self.failure_policy = setting["failure_policy"]
        self.response_format = setting.get("response_format")
        self.client = None
        self._init_client()

    def _init_client(self):
        api_key = self.setting.get("api_key") or os.getenv(self.setting.get("api_key_env", "DASHSCOPE_API_KEY"))
        if not api_key:
            self._handle_failure(ValueError(f"Missing {self.model} API key."))
            return

        from openai import OpenAI

        self.client = OpenAI(
            api_key=api_key,
            base_url=self.setting["base_url"],
            timeout=self.setting["timeout"],
        )

    def rerank(self, query_list, doc_list, batch_size=None, topk=None):
        if topk is None:
            topk = self.topk
        if isinstance(query_list, str):
            query_list = [query_list]
        if doc_list and not isinstance(doc_list[0], list):
            doc_list = [doc_list]

        assert len(query_list) == len(doc_list)
        if doc_list and topk > min([len(docs) for docs in doc_list]):
            warnings.warn("The number of doc returned by the retriever is less than the topk.")

        final_docs = []
        final_scores = []
        for query, docs in tqdm(list(zip(query_list, doc_list)), desc="LLM reranking process: "):
            try:
                ordered_indices = self._rank_one(query, docs)
            except Exception as exc:
                ordered_indices = self._handle_failure(exc, fallback=list(range(len(docs))))

            ordered_indices = ordered_indices[:topk]
            final_docs.append([docs[idx] for idx in ordered_indices])
            final_scores.append([float(len(ordered_indices) - rank) for rank, _ in enumerate(ordered_indices)])

        return final_docs, final_scores

    def _rank_one(self, query: str, docs: List[Any]) -> List[int]:
        if self.client is None:
            raise ValueError(f"{self.model} client is not initialized.")
        if not docs:
            return []

        candidates = self._build_candidates(docs)
        response = self._call_llm(query, candidates)
        parsed = self._parse_json(response)
        ranked_ids = self._extract_ranked_ids(parsed)
        return self._normalize_order(ranked_ids, candidates)

    def _call_llm(self, query: str, candidates: List[Dict[str, str]]) -> str:
        payload = {
            "question": query,
            "documents": candidates,
        }
        params = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": "Rank these candidate documents:\n"
                    + json.dumps(payload, ensure_ascii=False, indent=2),
                },
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
        return content

    def _build_candidates(self, docs: List[Any]) -> List[Dict[str, str]]:
        candidates = []
        for idx, doc in enumerate(docs):
            if isinstance(doc, str):
                title = ""
                content = doc
            else:
                title = str(doc.get("title", "") or "")
                content = str(doc.get("contents", "") or doc.get("text", "") or "")
            if len(content) > self.candidate_max_chars:
                content = content[: self.candidate_max_chars]
            candidates.append(
                {
                    "doc_id": f"doc_{idx + 1}",
                    "title": title,
                    "content": content,
                }
            )
        return candidates

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
    def _extract_ranked_ids(parsed: Dict[str, Any]) -> List[Any]:
        for key in ("ranked_doc_ids", "doc_ids", "ranking", "ranked_documents", "order"):
            value = parsed.get(key)
            if value is not None:
                if isinstance(value, list):
                    return value
                return [value]
        return []

    @staticmethod
    def _normalize_order(ranked_ids: List[Any], candidates: List[Dict[str, str]]) -> List[int]:
        valid_id_to_idx = {candidate["doc_id"]: idx for idx, candidate in enumerate(candidates)}
        ordered_indices = []
        seen = set()

        for item in ranked_ids:
            if isinstance(item, dict):
                item = item.get("doc_id") or item.get("id") or item.get("document_id")
            item = str(item).strip()
            if item.isdigit():
                item = f"doc_{item}"
            if item not in valid_id_to_idx or item in seen:
                continue
            seen.add(item)
            ordered_indices.append(valid_id_to_idx[item])

        ordered_indices.extend(idx for idx in range(len(candidates)) if idx not in ordered_indices)
        return ordered_indices

    def _handle_failure(self, exc: Exception, fallback=None):
        if self.failure_policy == "raise":
            raise exc

        warnings.warn(f"{self.model} LLM rerank failed, fallback to original retrieval order: {exc}", UserWarning)
        return fallback
