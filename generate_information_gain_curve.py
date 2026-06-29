import csv
import json
import os
import re
from typing import Any, Dict, Sequence

from flashrag.evaluator.metrics import F1_Score, Sub_ExactMatch


def generate_information_gain_curve(
    config,
    pipeline,
    dataset,
    prompt_template,
    run_name,
    doc_source,
    max_k=10,
    metric="f1",
    short_evidence_policy="hold",
    prediction_postprocess_func=None,
):
    """Generate an information gain curve over k retrieved/selected documents."""

    metric = metric.lower()
    if metric not in {"f1", "acc"}:
        raise ValueError("Information gain curve currently supports metric='f1' or metric='acc'.")
    if short_evidence_policy != "hold":
        raise ValueError("Information gain curve currently supports short_evidence_policy='hold' only.")

    max_k = int(max_k)
    if max_k < 0:
        raise ValueError("max_k must be >= 0.")

    save_dir = _config_value(config, "save_dir", "output")
    os.makedirs(save_dir, exist_ok=True)
    _prepare_matplotlib_cache_dir(save_dir)

    doc_lists = _get_curve_doc_lists(dataset, pipeline, doc_source, max_k)
    utility_by_sample = _compute_utility_by_sample(
        config=config,
        pipeline=pipeline,
        dataset=dataset,
        prompt_template=prompt_template,
        doc_lists=doc_lists,
        max_k=max_k,
        metric=metric,
        prediction_postprocess_func=prediction_postprocess_func,
    )
    sample_rows, summary_rows = _build_information_gain_rows(
        dataset=dataset,
        utility_by_sample=utility_by_sample,
        doc_lists=doc_lists,
        metric=metric,
    )

    safe_run_name = _safe_file_stem(run_name)
    file_stem = f"{safe_run_name}_information_gain_{metric}"
    csv_path = os.path.join(save_dir, f"{file_stem}.csv")
    json_path = os.path.join(save_dir, f"{file_stem}.json")
    png_path = os.path.join(save_dir, f"{file_stem}.png")

    _write_summary_csv(csv_path, summary_rows, metric)
    _write_json(
        json_path,
        {
            "run_name": run_name,
            "metric": metric,
            "doc_source": doc_source,
            "max_k": max_k,
            "short_evidence_policy": short_evidence_policy,
            "summary": summary_rows,
            "per_sample": sample_rows,
        },
    )
    _plot_information_gain_curve(png_path, summary_rows, run_name, metric)

    print("=== information gain curve ===")
    print(f"CSV: {csv_path}")
    print(f"JSON: {json_path}")
    print(f"PNG: {png_path}")

    return {
        "csv_path": csv_path,
        "json_path": json_path,
        "png_path": png_path,
        "summary": summary_rows,
    }


def _config_value(config, key, default=None):
    if isinstance(config, dict):
        return config.get(key, default)
    value = config[key]
    return default if value is None else value


def _get_curve_doc_lists(dataset, pipeline, doc_source, max_k):
    if doc_source == "retrieval_result":
        current_doc_lists = _read_doc_source_from_dataset(dataset, doc_source)
        if _min_doc_count(current_doc_lists) < max_k:
            if getattr(pipeline, "retriever", None) is None:
                raise ValueError("pipeline.retriever is required to build the retrieval_result IG curve.")
            return pipeline.retriever.batch_search(dataset.question, num=max_k)
        return current_doc_lists

    if doc_source == "selected_evidence":
        return _read_doc_source_from_dataset(dataset, doc_source)

    raise ValueError(f"Unsupported doc_source: {doc_source}")


def _read_doc_source_from_dataset(dataset, doc_source):
    doc_lists = []
    for item_idx, item in enumerate(dataset):
        if doc_source not in item.output:
            raise ValueError(f"Missing `{doc_source}` in dataset item {item_idx}.")
        docs = item.output[doc_source]
        if docs is None:
            docs = []
        doc_lists.append(list(docs))
    return doc_lists


def _min_doc_count(doc_lists):
    if not doc_lists:
        return 0
    return min(len(docs) for docs in doc_lists)


def _compute_utility_by_sample(
    config,
    pipeline,
    dataset,
    prompt_template,
    doc_lists,
    max_k,
    metric,
    prediction_postprocess_func=None,
):
    scorer, score_func = _build_metric_scorer(config, metric)
    golden_answers_list = scorer.get_dataset_answer(dataset)
    utility_by_sample = [[None for _ in range(max_k + 1)] for _ in range(len(dataset))]

    for k in range(max_k + 1):
        prompt_items = []
        for sample_idx, docs in enumerate(doc_lists):
            if k > 0 and k > len(docs):
                utility_by_sample[sample_idx][k] = utility_by_sample[sample_idx][k - 1]
                continue

            selected_docs = docs[:k]
            prompt_items.append(
                (
                    sample_idx,
                    prompt_template.get_string(
                        question=dataset[sample_idx].question,
                        retrieval_result=selected_docs,
                    ),
                )
            )

        if not prompt_items:
            continue

        prompts = [prompt for _, prompt in prompt_items]
        predictions = pipeline.generator.generate(prompts)

        for (sample_idx, _), prediction in zip(prompt_items, predictions):
            if prediction_postprocess_func is not None:
                prediction = prediction_postprocess_func(prediction)
            score = score_func(prediction, golden_answers_list[sample_idx])
            utility_by_sample[sample_idx][k] = float(score)

    return utility_by_sample


def _build_metric_scorer(config, metric):
    if metric == "f1":
        scorer = F1_Score(config)
        return scorer, lambda prediction, golden_answers: scorer.token_level_scores(prediction, golden_answers)["f1"]
    if metric == "acc":
        scorer = Sub_ExactMatch(config)
        return scorer, scorer.calculate_sub_em
    raise ValueError(f"Unsupported metric: {metric}")


def _build_information_gain_rows(dataset, utility_by_sample, doc_lists, metric):
    sample_rows = []
    for sample_idx, utility_values in enumerate(utility_by_sample):
        if any(value is None for value in utility_values):
            raise ValueError(f"Failed to compute utility for dataset item {sample_idx}.")

        cumulative_values = [0.0]
        baseline = utility_values[0]
        for k in range(1, len(utility_values)):
            cumulative_values.append(utility_values[k] - baseline)

        sample_rows.append(
            {
                "sample_index": sample_idx,
                "id": dataset[sample_idx].id,
                "available_doc_count": len(doc_lists[sample_idx]),
                f"utility_{metric}": utility_values,
                "cumulative_ig": cumulative_values,
            }
        )

    summary_rows = []
    max_k = len(utility_by_sample[0]) - 1 if utility_by_sample else 0
    for k in range(max_k + 1):
        utility_values = [row[f"utility_{metric}"][k] for row in sample_rows]
        cumulative_values = [row["cumulative_ig"][k] for row in sample_rows]
        generated_count = sum(1 for docs in doc_lists if k == 0 or k <= len(docs))
        summary_rows.append(
            {
                "k": k,
                f"avg_utility_{metric}": _mean(utility_values),
                "avg_cumulative_ig": _mean(cumulative_values),
                "sample_count": len(sample_rows),
                "generated_count": generated_count,
            }
        )
    return sample_rows, summary_rows


def _mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _write_summary_csv(path, summary_rows, metric):
    fieldnames = [
        "k",
        f"avg_utility_{metric}",
        "avg_cumulative_ig",
        "sample_count",
        "generated_count",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)


def _write_json(path, payload: Dict[str, Any]):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4, ensure_ascii=False)


def _prepare_matplotlib_cache_dir(save_dir):
    cache_dir = os.path.join(save_dir, ".matplotlib")
    os.makedirs(cache_dir, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", cache_dir)


def _plot_information_gain_curve(path, summary_rows, run_name, metric):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x_values = [row["k"] for row in summary_rows]
    cumulative_values = [row["avg_cumulative_ig"] for row in summary_rows]

    fig, axis = plt.subplots(figsize=(10, 6))
    line_color = "#F58518"

    axis.plot(
        x_values,
        cumulative_values,
        color=line_color,
        marker="o",
        linewidth=2.2,
        label="Average Cumulative Information Gain",
    )
    axis.axhline(0, color="#666666", linewidth=0.8)
    axis.set_xlabel("k")
    axis.set_ylabel("Average Cumulative Information Gain")
    axis.set_xticks(x_values)
    axis.set_title(f"{run_name} Cumulative Information Gain Curve ({metric.upper()})")
    axis.grid(axis="y", linestyle="--", alpha=0.25)
    axis.legend(loc="best", frameon=False)

    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _safe_file_stem(value):
    value = str(value).strip().lower()
    value = re.sub(r"[^a-z0-9._-]+", "_", value)
    return value.strip("._-") or "run"
