import copy
import hashlib
from typing import Any, Dict, List, Optional


class EvidencePoolBuilder:
    """Merge branch retrieval results into a deduplicated evidence pool."""

    def __init__(self, setting: Optional[Dict[str, Any]] = None):
        setting = setting or {}
        self.score_aggregation = setting.get("score_aggregation", "max")
        self.final_topk = setting.get("final_topk", None)

    def build(self, branch_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        doc_map = {}
        first_seen = 0

        for branch_result in branch_results:
            branch_id = branch_result["branch_id"]
            branch_query = branch_result["branch_query"]
            documents = branch_result.get("documents", [])
            scores = branch_result.get("scores", [])

            for rank, doc in enumerate(documents, start=1):
                score = self._normalize_score(scores[rank - 1] if rank - 1 < len(scores) else None)
                doc_key = self._get_doc_key(doc)
                source = {
                    "branch_id": branch_id,
                    "branch_query": branch_query,
                    "branch_rank": rank,
                    "dense_similarity": score,
                }

                if doc_key not in doc_map:
                    merged_doc = copy.deepcopy(doc)
                    merged_doc["sources"] = [source]
                    merged_doc["_first_seen"] = first_seen
                    first_seen += 1
                    doc_map[doc_key] = merged_doc
                else:
                    doc_map[doc_key]["sources"].append(source)

        evidence_pool = [self._finalize_doc(doc) for doc in doc_map.values()]
        evidence_pool.sort(key=self._sort_key)

        if self.final_topk is not None:
            final_topk = int(self.final_topk)
            if final_topk > 0:
                evidence_pool = evidence_pool[:final_topk]

        for doc in evidence_pool:
            doc.pop("_first_seen", None)
        return evidence_pool

    def _finalize_doc(self, doc: Dict[str, Any]) -> Dict[str, Any]:
        sources = doc["sources"]
        scores = [source["dense_similarity"] for source in sources if source["dense_similarity"] is not None]
        doc["hit_count"] = len(sources)
        doc["dense_similarity"] = self._aggregate_scores(scores)
        doc["best_branch_rank"] = min(source["branch_rank"] for source in sources)
        return doc

    def _aggregate_scores(self, scores: List[float]) -> Optional[float]:
        if not scores:
            return None
        if self.score_aggregation == "avg":
            return sum(scores) / len(scores)
        if self.score_aggregation == "sum":
            return sum(scores)
        return max(scores)

    @staticmethod
    def _sort_key(doc: Dict[str, Any]):
        score = doc["dense_similarity"]
        score_for_sort = score if score is not None else float("-inf")
        return (
            -doc["hit_count"],
            -score_for_sort,
            doc["best_branch_rank"],
            doc["_first_seen"],
        )

    @staticmethod
    def _normalize_score(score: Any) -> Optional[float]:
        if score is None:
            return None
        try:
            return float(score)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _get_doc_key(doc: Dict[str, Any]) -> str:
        doc_id = doc.get("id")
        if doc_id is not None:
            return f"id::{doc_id}"

        contents = doc.get("contents", "")
        if not contents:
            contents = f"{doc.get('title', '')}\n{doc.get('text', '')}"
        digest = hashlib.sha1(contents.encode("utf-8", errors="ignore")).hexdigest()
        return f"contents::{digest}"
