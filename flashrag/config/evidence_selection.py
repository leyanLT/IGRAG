from typing import Any, Dict, Optional


DATASET_EVIDENCE_SELECTION_LIMITS = {
    "nq": {"min_selected": 1, "max_selected": 4},
    "hotpotqa": {"min_selected": 2, "max_selected": 4},
    "2wikimultihopqa": {"min_selected": 2, "max_selected": 4},
    "multihoprag": {"min_selected": 2, "max_selected": 3},
}


def get_dataset_evidence_selection_limits(dataset_name: Any) -> Optional[Dict[str, int]]:
    normalized_name = str(dataset_name).strip().lower()
    limits = DATASET_EVIDENCE_SELECTION_LIMITS.get(normalized_name)
    if limits is None:
        return None
    return dict(limits)


def apply_dataset_evidence_selection_limits(setting: Dict[str, Any], dataset_name: Any) -> Dict[str, Any]:
    limits = get_dataset_evidence_selection_limits(dataset_name)
    if limits is not None:
        setting.update(limits)
    return setting
