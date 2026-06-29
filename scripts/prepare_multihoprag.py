#!/usr/bin/env python3
"""Convert the original MultiHopRAG files into FlashRAG-compatible JSONL files."""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


DEFAULT_DATASET_DIR = "/root/autodl-tmp/FlashRAG_Data/MultiHopRAG"


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def as_answer_list(answer: Any) -> List[str]:
    if answer is None:
        return []
    if isinstance(answer, list):
        return [str(item) for item in answer if item is not None]
    return [str(answer)]


def build_contents(doc: Dict[str, Any]) -> str:
    title = doc.get("title") or ""
    source = doc.get("source") or ""
    category = doc.get("category") or ""
    published_at = doc.get("published_at") or ""
    body = doc.get("body") or ""

    meta = " | ".join(str(item) for item in [source, category, published_at] if item)
    parts = [str(item) for item in [title, meta, body] if item]
    return "\n".join(parts)


def iter_dataset_records(items: Iterable[Dict[str, Any]]) -> Iterable[Dict[str, Any]]:
    for idx, item in enumerate(items):
        yield {
            "id": str(idx),
            "question": item.get("query", ""),
            "golden_answers": as_answer_list(item.get("answer")),
            "metadata": {
                "question_type": item.get("question_type"),
                "evidence_list": item.get("evidence_list", []),
            },
        }


def iter_corpus_records(docs: Iterable[Dict[str, Any]]) -> Iterable[Dict[str, Any]]:
    for idx, doc in enumerate(docs):
        yield {
            "id": str(idx),
            "contents": build_contents(doc),
            "title": doc.get("title"),
            "author": doc.get("author"),
            "source": doc.get("source"),
            "category": doc.get("category"),
            "published_at": doc.get("published_at"),
            "url": doc.get("url"),
        }


def write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> int:
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare MultiHopRAG for FlashRAG evaluation."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path(DEFAULT_DATASET_DIR),
        help="Directory containing MultiHopRAG.json and corpus.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to --dataset-dir.",
    )
    parser.add_argument(
        "--split",
        default="dev",
        help="Output split name for the QA file, for example dev/test/train.",
    )
    parser.add_argument(
        "--qa-file",
        default="MultiHopRAG.json",
        help="Original QA filename inside --dataset-dir.",
    )
    parser.add_argument(
        "--corpus-file",
        default="corpus.json",
        help="Original corpus filename inside --dataset-dir.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset_dir.expanduser().resolve()
    output_dir = (args.output_dir or dataset_dir).expanduser().resolve()

    qa_path = dataset_dir / args.qa_file
    corpus_path = dataset_dir / args.corpus_file
    dataset_output_path = output_dir / f"{args.split}.jsonl"
    corpus_output_path = output_dir / "corpus.jsonl"

    if not qa_path.exists():
        raise FileNotFoundError(f"QA file not found: {qa_path}")
    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus file not found: {corpus_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    qa_data = load_json(qa_path)
    corpus_data = load_json(corpus_path)

    if not isinstance(qa_data, list):
        raise TypeError(f"Expected QA data to be a list, got {type(qa_data).__name__}")
    if not isinstance(corpus_data, list):
        raise TypeError(f"Expected corpus data to be a list, got {type(corpus_data).__name__}")

    dataset_count = write_jsonl(dataset_output_path, iter_dataset_records(qa_data))
    corpus_count = write_jsonl(corpus_output_path, iter_corpus_records(corpus_data))

    print(f"Wrote {dataset_count} QA records to {dataset_output_path}")
    print(f"Wrote {corpus_count} corpus records to {corpus_output_path}")


if __name__ == "__main__":
    main()
