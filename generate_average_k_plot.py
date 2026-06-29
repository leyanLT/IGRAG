import csv
import json
import os
import re
from typing import Any, Dict, List


def generate_average_k_plot(config, dataset, run_name, bin_size=3, bin_start=1, k_source="selected_count"):
    """Generate a compact average-k summary for selected evidence count k."""

    bin_size = int(bin_size)
    if bin_size <= 0:
        raise ValueError("bin_size must be > 0.")
    if k_source not in {"selected_count", "retrieval_result_count"}:
        raise ValueError("k_source must be 'selected_count' or 'retrieval_result_count'.")

    save_dir = _config_value(config, "save_dir", "output")
    os.makedirs(save_dir, exist_ok=True)
    _prepare_matplotlib_cache_dir(save_dir)

    k_values = _extract_k_values(dataset, k_source)
    if not k_values:
        raise ValueError(f"No valid k values found from source: {k_source}.")

    summary = _build_summary(k_values)

    safe_run_name = _safe_file_stem(run_name)
    file_stem = f"{safe_run_name}_average_k"
    csv_path = os.path.join(save_dir, f"{file_stem}.csv")
    json_path = os.path.join(save_dir, f"{file_stem}.json")
    png_path = os.path.join(save_dir, f"{file_stem}.png")

    _write_summary_csv(csv_path, summary)
    _write_json(
        json_path,
        {
            "run_name": run_name,
            "k_source": k_source,
            "k_definition": _k_definition(k_source),
            "summary": summary,
        },
    )
    _plot_average_k(png_path, summary, run_name)

    print("=== average k ===")
    print(f"Average k: {summary['avg_k']:.4f}")
    print(f"CSV: {csv_path}")
    print(f"JSON: {json_path}")
    print(f"PNG: {png_path}")

    return {
        "csv_path": csv_path,
        "json_path": json_path,
        "png_path": png_path,
        "summary": summary,
    }


def _config_value(config, key, default=None):
    if isinstance(config, dict):
        return config.get(key, default)
    value = config[key]
    return default if value is None else value


def _extract_k_values(dataset, k_source: str) -> List[int]:
    k_values = []
    for item_idx, item in enumerate(dataset):
        output = _item_output(item)

        if k_source == "retrieval_result_count":
            if "retrieval_result" not in output:
                raise ValueError(f"Missing `retrieval_result` in dataset item {item_idx}.")
            retrieval_result = output["retrieval_result"]
            if retrieval_result is None:
                retrieval_result = []
            k_values.append(len(retrieval_result))
            continue

        if "evidence_selection_trace" not in output:
            raise ValueError(f"Missing `evidence_selection_trace` in dataset item {item_idx}.")

        trace = output["evidence_selection_trace"]
        if not isinstance(trace, dict):
            raise ValueError(f"`evidence_selection_trace` must be a dict in dataset item {item_idx}.")
        if "selected_count" not in trace:
            raise ValueError(f"Missing `selected_count` in dataset item {item_idx}.")

        selected_count = trace["selected_count"]
        if isinstance(selected_count, bool) or not isinstance(selected_count, int):
            raise ValueError(f"`selected_count` must be an integer in dataset item {item_idx}.")
        if selected_count < 0:
            raise ValueError(f"`selected_count` must be >= 0 in dataset item {item_idx}.")

        k_values.append(selected_count)
    return k_values


def _k_definition(k_source: str) -> str:
    if k_source == "selected_count":
        return "output.evidence_selection_trace.selected_count"
    if k_source == "retrieval_result_count":
        return "len(output.retrieval_result)"
    raise ValueError(f"Unsupported k_source: {k_source}")


def _item_output(item):
    if isinstance(item, dict):
        return item.get("output", {})
    output = getattr(item, "output", None)
    if output is None:
        raise ValueError("Dataset item does not expose an output field.")
    return output


def _build_summary(k_values: List[int]) -> Dict[str, Any]:
    return {
        "avg_k": float(sum(k_values) / len(k_values)),
        "valid_sample_count": len(k_values),
    }


def _write_summary_csv(path, summary: Dict[str, Any]):
    fieldnames = [
        "avg_k",
        "valid_sample_count",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(
            {
                "avg_k": summary["avg_k"],
                "valid_sample_count": summary["valid_sample_count"],
            }
        )


def _write_json(path, payload: Dict[str, Any]):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4, ensure_ascii=False)


def _prepare_matplotlib_cache_dir(save_dir):
    cache_dir = os.path.join(save_dir, ".matplotlib")
    os.makedirs(cache_dir, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", cache_dir)


def _plot_average_k(path, summary: Dict[str, Any], run_name):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    avg_k = summary["avg_k"]
    fig, axis = plt.subplots(figsize=(7, 4))
    axis.axis("off")
    axis.text(
        0.5,
        0.58,
        f"{avg_k:.2f}",
        ha="center",
        va="center",
        fontsize=52,
        fontweight="bold",
        color="#4C78A8",
    )
    axis.text(
        0.5,
        0.35,
        "Average k",
        ha="center",
        va="center",
        fontsize=18,
        color="#333333",
    )
    axis.set_title(f"{run_name} Average k", fontsize=16, pad=20)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _safe_file_stem(value):
    value = str(value).strip().lower()
    value = re.sub(r"[^a-z0-9._-]+", "_", value)
    return value.strip("._-") or "run"
