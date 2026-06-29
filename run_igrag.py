from flashrag.config import Config
from flashrag.utils import get_dataset
from flashrag.pipeline import IGRAGPipeline
from flashrag.prompt import PromptTemplate
from generate_information_gain_curve import generate_information_gain_curve
from generate_average_k_plot import generate_average_k_plot


BGE_RETRIEVER_NAME = "bge-large-en-v1.5"
BGE_RETRIEVER_PATH = "/root/autodl-tmp/models/BAAI/bge-large-en-v1.5"


config_dict = {
    "gpu_id": "0",
    "framework": "vllm",

    "data_dir": "/root/autodl-tmp/FlashRAG_Data",
    "dataset_name": "hotpotqa",
    "split": ["dev"],
    "test_sample_num": 1000,
    "random_sample": True,

    "corpus_path": "/root/autodl-tmp/FlashRAG_Data/retrieval-corpus/wiki18_100w.jsonl",
    "index_path": "/root/autodl-tmp/FlashRAG_Data/indexes/bge_large_en_wiki18_100w_sharded",

    "model2path": {
        BGE_RETRIEVER_NAME: BGE_RETRIEVER_PATH,
        "llama-3.1-8b-instruct": "/root/autodl-tmp/models/Llama-3.1-8B-Instruct",
    },

    "retrieval_method": BGE_RETRIEVER_NAME,
    "retrieval_pooling_method": "cls",
    "generator_model": "llama-3.1-8b-instruct",

    "retrieval_topk": 5,
    "use_query_rewrite": True,
    "query_rewrite_setting": {
        "model": "qwen-turbo",
        "api_key": "sk-356858d231bc4085a2cd1bc1f6648c00",
        "api_key_env": "DASHSCOPE_API_KEY",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "max_branches": 3,
        "original_topk": 6,
        "branch_topk": 6,
        "temperature": 0,
        "max_tokens": 512,
        "timeout": 60,
        "failure_policy": "fallback",
    },
    "evidence_pool_setting": {
        "score_aggregation": "max",
        "final_topk": None,
    },
    "evidence_selection_stop_setting": {
        "enabled": True,
        "model": "qwen2.5-evidence-stop-sft",
        "api_key": "EMPTY",
        "base_url": "http://127.0.0.1:8001/v1",
        "timeout": 120,
        "temperature": 0,
        "max_tokens": 1024,
        "candidate_max_chars": 800,
        "failure_policy": "fallback",
        "response_format": None,
    },
    "generator_batch_size": 4,
    "generator_max_input_len": 2048,
    "generation_params": {
        "do_sample": False,
        "max_tokens": 64,
    },

    "metrics": ["em", "f1", "acc",  "context_precision@k", "recall@k",  "ndcg@k", "mrr@k"],
    "save_intermediate_data": True,
    "save_note": "igrag_hotpotqa_dev_sft_stopper",
    "enable_information_gain_curve": False,
    "enable_average_k_plot": True,
}


def main():
    config = Config(config_dict=config_dict)

    all_split = get_dataset(config)
    split_name = config["split"][0]
    eval_data = all_split[split_name]

    prompt_template = PromptTemplate(
        config,
        system_prompt="Answer the question based on the given document. Only give me the answer and do not output any other words.\nThe following are given documents.\n\n{reference}",
        user_prompt="Question: {question}\nAnswer:",
    )

    pipeline = IGRAGPipeline(config, prompt_template=prompt_template)
    output_dataset = pipeline.run(eval_data, do_eval=True)

    print("=== prediction ===")
    print(output_dataset.pred[:5])

    if config["enable_information_gain_curve"]:
        generate_information_gain_curve(
            config=config,
            pipeline=pipeline,
            dataset=output_dataset,
            prompt_template=prompt_template,
            run_name="igrag",
            doc_source="selected_evidence",
            max_k=10,
            metric="f1",
            short_evidence_policy="hold",
        )

    if config["enable_average_k_plot"]:
        generate_average_k_plot(
            config=config,
            dataset=output_dataset,
            run_name="igrag",
        )


if __name__ == "__main__":
    main()
