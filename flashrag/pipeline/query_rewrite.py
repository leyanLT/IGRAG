import json
import os
import re
import warnings
from typing import Any, Dict, List


class QwenTurboQueryRewriter:
    """Rewrite retrieval queries with qwen-turbo through the OpenAI-compatible API."""

    DEFAULT_SETTING = {
        "model": "qwen-turbo",
        "api_key": None,
        "api_key_env": "DASHSCOPE_API_KEY",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "max_branches": 3,
        "original_topk": 5,
        "branch_topk": 3,
        "temperature": 0,
        "max_tokens": 512,
        "timeout": 60,
        "failure_policy": "raise",
        "response_format": {"type": "json_object"},
    }

    SYSTEM_PROMPT = """You are a query analysis and rewriting module for dense retrieval.
The downstream retriever is BAAI/bge-large-en-v1.5. The retriever already prepends the official query instruction, so never include "Represent this sentence for searching relevant passages:" in rewritten queries.

Your job:
1. Rewrite up to 3 retrieval query branches.
2. Each branch must serve the final goal of the original query, not an isolated sub-question.
3. Together, the original query and rewritten branches should cover all information needs.
4. Rewritten queries should be short, explicit, entity-rich, and suitable for English dense retrieval.

Return strict JSON only, with exactly these keys:
{
  "rewritten_queries": ["..."]
}
"""

    def __init__(self, config):
        setting = dict(self.DEFAULT_SETTING)
        setting.update(config["query_rewrite_setting"] or {})
        self.setting = setting

        self.model = setting["model"]
        self.max_branches = int(setting["max_branches"])
        self.original_topk = int(setting["original_topk"])
        self.branch_topk = int(setting["branch_topk"])
        self.temperature = setting["temperature"]
        self.max_tokens = int(setting["max_tokens"])
        self.failure_policy = setting["failure_policy"]
        self.response_format = setting.get("response_format")

        api_key = setting.get("api_key") or os.getenv(setting.get("api_key_env", "DASHSCOPE_API_KEY"))
        if not api_key:
            self.client = None
            self._handle_failure(ValueError("Missing qwen-turbo API key."), query=None)
        else:
            from openai import OpenAI

            self.client = OpenAI(
                api_key=api_key,
                base_url=setting["base_url"],
                timeout=setting["timeout"],
            )

    def batch_rewrite(self, queries: List[str]) -> List[Dict[str, Any]]:
        return [self.rewrite(query) for query in queries]

    def rewrite(self, query: str) -> Dict[str, Any]:
        try:
            raw_response = self._call_qwen(query)
            parsed = self._parse_json(raw_response)
            return self._normalize_result(query, parsed)
        except Exception as exc:
            return self._handle_failure(exc, query=query)

    def _call_qwen(self, query: str) -> str:
        if self.client is None:
            raise ValueError("qwen-turbo client is not initialized.")

        params = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": f"Original query:\n{query}"},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if self.response_format:
            params["response_format"] = self.response_format

        response = self.client.chat.completions.create(**params)
        if not response.choices:
            raise ValueError("No choices returned from qwen-turbo.")
        content = response.choices[0].message.content
        if not content:
            raise ValueError("Empty response returned from qwen-turbo.")
        return content

    @staticmethod
    def _parse_json(text: str) -> Dict[str, Any]:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, flags=re.DOTALL)
            if match is None:
                raise
            return json.loads(match.group(0))

    def _normalize_result(self, query: str, parsed: Dict[str, Any]) -> Dict[str, Any]:
        rewritten_queries = self._normalize_string_list(parsed.get("rewritten_queries", []))

        unique_rewrites = []
        seen = {query.strip().lower()}
        for rewritten_query in rewritten_queries:
            key = rewritten_query.lower()
            if key in seen:
                continue
            seen.add(key)
            unique_rewrites.append(rewritten_query)
            if len(unique_rewrites) >= self.max_branches:
                break

        branches = [{"branch_id": "original", "query": query, "source": "original"}]
        for idx, rewritten_query in enumerate(unique_rewrites, start=1):
            branches.append(
                {
                    "branch_id": f"rewrite_{idx}",
                    "query": rewritten_query,
                    "source": self.model,
                }
            )

        return {
            "original_query": query,
            "rewritten_queries": unique_rewrites,
            "branches": branches,
        }

    @staticmethod
    def _normalize_string_list(value: Any) -> List[str]:
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, list):
            return []

        normalized = []
        for item in value:
            if isinstance(item, dict):
                item = item.get("query") or item.get("text") or ""
            item = str(item).strip()
            if item:
                normalized.append(item)
        return normalized

    def _handle_failure(self, exc: Exception, query: str = None) -> Dict[str, Any]:
        if self.failure_policy == "raise":
            raise exc

        warnings.warn(f"qwen-turbo query rewrite failed, fallback to original query: {exc}", UserWarning)
        return {
            "original_query": query,
            "rewritten_queries": [],
            "branches": [{"branch_id": "original", "query": query, "source": "original"}],
            "error": str(exc),
        }
