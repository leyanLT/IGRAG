from flashrag.evaluator import Evaluator
from flashrag.dataset.utils import split_dataset, merge_dataset
from flashrag.utils import get_retriever, get_generator, get_refiner, get_judger
from flashrag.prompt import PromptTemplate
from flashrag.pipeline.evidence_pool import EvidencePoolBuilder
from flashrag.pipeline.evidence_selection_stop import EvidenceSelectionStopper
from flashrag.pipeline.query_rewrite import QwenTurboQueryRewriter


class BasicPipeline:
    """Base object of all pipelines. A pipeline includes the overall process of RAG.
    If you want to implement a pipeline, you should inherit this class.
    """

    def __init__(self, config, prompt_template=None):
        self.config = config
        self.device = config["device"]
        self.retriever = None
        self.evaluator = Evaluator(config)
        self.save_retrieval_cache = config["save_retrieval_cache"]
        if prompt_template is None:
            prompt_template = PromptTemplate(config)
        self.prompt_template = prompt_template

    def run(self, dataset):
        """The overall inference process of a RAG framework."""
        pass

    def evaluate(self, dataset, do_eval=True, pred_process_fun=None):
        """The evaluation process after finishing overall generation"""

        if pred_process_fun is not None:
            dataset = pred_process_fun(dataset)

        if do_eval:
            # evaluate & save result
            eval_result = self.evaluator.evaluate(dataset)
            print(eval_result)

        # save retrieval cache
        if self.save_retrieval_cache:
            self.retriever._save_cache()

        return dataset


class SequentialPipeline(BasicPipeline):
    def __init__(self, config, prompt_template=None, retriever=None, generator=None):
        """
        inference stage:
            query -> pre-retrieval -> retriever -> post-retrieval -> generator
        """

        super().__init__(config, prompt_template)
        if generator is None:
            self.generator = get_generator(config)
        else:
            self.generator = generator

        if retriever is None:
            self.retriever = get_retriever(config)
        else:
            self.retriever = retriever

        # TODO: add rewriter module

        self.use_fid = config["use_fid"]

        if config["refiner_name"] is not None:
            self.refiner = get_refiner(config, self.retriever, self.generator)
        else:
            self.refiner = None

    def naive_run(self, dataset, do_eval=True, pred_process_fun=None):
        # direct generation without RAG
        input_prompts = [self.prompt_template.get_string(question=q) for q in dataset.question]
        dataset.update_output("prompt", input_prompts)

        pred_answer_list = self.generator.generate(input_prompts)
        dataset.update_output("pred", pred_answer_list)

        dataset = self.evaluate(dataset, do_eval=do_eval, pred_process_fun=pred_process_fun)
        return dataset

    def run(self, dataset, do_eval=True, pred_process_fun=None):
        input_query = dataset.question
        retrieval_results = self.retriever.batch_search(input_query)
        dataset.update_output("retrieval_result", retrieval_results)

        if self.refiner:
            input_prompt_flag = self.refiner.input_prompt_flag
            if "llmlingua" in self.refiner.name and input_prompt_flag:
                # input prompt
                input_prompts = [
                    self.prompt_template.get_string(question=q, retrieval_result=r)
                    for q, r in zip(dataset.question, dataset.retrieval_result)
                ]
                dataset.update_output("prompt", input_prompts)
                input_prompts = self.refiner.batch_run(dataset)
            else:
                # input retrieval docs
                refine_results = self.refiner.batch_run(dataset)
                dataset.update_output("refine_result", refine_results)
                input_prompts = [
                    self.prompt_template.get_string(question=q, formatted_reference=r)
                    for q, r in zip(dataset.question, refine_results)
                ]

        else:
            if not self.use_fid:
                input_prompts = [
                    self.prompt_template.get_string(question=q, retrieval_result=r)
                    for q, r in zip(dataset.question, dataset.retrieval_result)
                ]

        if self.use_fid:
            print("Use FiD generation")
            input_prompts = []
            for item in dataset:
                q = item.question
                docs = item.retrieval_result
                input_prompts.append([q + " " + doc['contents'] for doc in docs])
        dataset.update_output("prompt", input_prompts)

        # delete used refiner to release memory
        if self.refiner:
            del self.refiner
        pred_answer_list = self.generator.generate(input_prompts)
        dataset.update_output("pred", pred_answer_list)

        dataset = self.evaluate(dataset, do_eval=do_eval, pred_process_fun=pred_process_fun)

        return dataset


class IGRAGPipeline(SequentialPipeline):
    """An extensible standard RAG pipeline with stable stage hooks."""

    def __init__(self, config, prompt_template=None, retriever=None, generator=None):
        super().__init__(config, prompt_template, retriever, generator)
        self.use_query_rewrite = bool(config["use_query_rewrite"])
        self.query_rewriter = QwenTurboQueryRewriter(config) if self.use_query_rewrite else None
        self.evidence_pool_builder = (
            EvidencePoolBuilder(config["evidence_pool_setting"]) if self.use_query_rewrite else None
        )
        evidence_selection_setting = config["evidence_selection_stop_setting"] or {}
        self.use_evidence_selection_stop = bool(evidence_selection_setting.get("enabled", False))
        self.evidence_selector = (
            EvidenceSelectionStopper(config, self.retriever) if self.use_evidence_selection_stop else None
        )
        self._reset_stage_state()

    def _reset_stage_state(self):
        self._stage_state = {
            "enable_query_rewrite": False,
            "query_list": None,
            "query_rewrite_result": None,
            "branch_retrieval_result": None,
            "evidence_pool": None,
            "selected_evidence": None,
            "evidence_selection_trace": None,
            "retrieval_postprocess": None,
            "retrieval_postprocess_type": None,
            "retrieval_postprocess_label": None,
            "input_prompts": None,
        }

    def _ensure_refiner(self):
        if self.config["refiner_name"] is not None and self.refiner is None:
            self.refiner = get_refiner(self.config, self.retriever, self.generator)

    def _release_refiner(self):
        if self.refiner is not None:
            self.refiner = None

    def _preprocess_query(self, dataset):
        """Hook for query rewriting or normalization before retrieval."""
        self._stage_state["query_list"] = dataset.question
        if self.use_query_rewrite and self._stage_state.get("enable_query_rewrite"):
            query_rewrite_results = self.query_rewriter.batch_rewrite(dataset.question)
            dataset.update_output("query_rewrite_result", query_rewrite_results)
            self._stage_state["query_rewrite_result"] = query_rewrite_results
        return dataset

    def _retrieve(self, dataset):
        if self.use_query_rewrite and self._stage_state.get("enable_query_rewrite"):
            return self._retrieve_with_query_rewrite(dataset)

        query_list = self._stage_state.get("query_list") or dataset.question
        retrieval_results = self.retriever.batch_search(query_list)
        dataset.update_output("retrieval_result", retrieval_results)
        return dataset

    def _retrieve_with_query_rewrite(self, dataset):
        query_rewrite_results = self._stage_state.get("query_rewrite_result")
        if query_rewrite_results is None:
            query_rewrite_results = self.query_rewriter.batch_rewrite(dataset.question)
            dataset.update_output("query_rewrite_result", query_rewrite_results)
            self._stage_state["query_rewrite_result"] = query_rewrite_results

        original_queries = [item["original_query"] for item in query_rewrite_results]
        original_results, original_scores = self.retriever.batch_search(
            original_queries,
            num=self.query_rewriter.original_topk,
            return_score=True,
        )

        branch_retrieval_results = []
        rewrite_queries = []
        rewrite_metadata = []

        for item_idx, rewrite_result in enumerate(query_rewrite_results):
            item_branch_results = [
                {
                    "branch_id": "original",
                    "branch_query": rewrite_result["original_query"],
                    "topk": self.query_rewriter.original_topk,
                    "documents": original_results[item_idx],
                    "scores": original_scores[item_idx],
                }
            ]
            branch_retrieval_results.append(item_branch_results)

            for branch in rewrite_result["branches"]:
                if branch["branch_id"] == "original":
                    continue
                rewrite_queries.append(branch["query"])
                rewrite_metadata.append((item_idx, branch))

        if rewrite_queries:
            rewrite_results, rewrite_scores = self.retriever.batch_search(
                rewrite_queries,
                num=self.query_rewriter.branch_topk,
                return_score=True,
            )
            for (item_idx, branch), documents, scores in zip(rewrite_metadata, rewrite_results, rewrite_scores):
                branch_retrieval_results[item_idx].append(
                    {
                        "branch_id": branch["branch_id"],
                        "branch_query": branch["query"],
                        "topk": self.query_rewriter.branch_topk,
                        "documents": documents,
                        "scores": scores,
                    }
                )

        evidence_pools = [self.evidence_pool_builder.build(item_results) for item_results in branch_retrieval_results]
        dataset.update_output("branch_retrieval_result", branch_retrieval_results)
        dataset.update_output("evidence_pool", evidence_pools)
        dataset.update_output("retrieval_result", evidence_pools)

        self._stage_state["branch_retrieval_result"] = branch_retrieval_results
        self._stage_state["evidence_pool"] = evidence_pools
        return dataset

    def _select_evidence(self, dataset):
        if self.evidence_selector is None:
            return dataset
        if len(dataset) == 0:
            return dataset

        evidence_pools = []
        has_evidence_pool_output = True
        for item in dataset:
            if "evidence_pool" in item.output:
                evidence_pools.append(item.evidence_pool)
            else:
                has_evidence_pool_output = False
                evidence_pools.append(item.retrieval_result)

        if not has_evidence_pool_output:
            dataset.update_output("evidence_pool", evidence_pools)

        selected_evidence = []
        evidence_selection_trace = []
        for query, evidence_pool in zip(dataset.question, evidence_pools):
            selected_docs, trace = self.evidence_selector.select(query, evidence_pool)
            selected_evidence.append(selected_docs)
            evidence_selection_trace.append(trace)

        dataset.update_output("selected_evidence", selected_evidence)
        dataset.update_output("evidence_selection_trace", evidence_selection_trace)
        dataset.update_output("retrieval_result", selected_evidence)

        self._stage_state["evidence_pool"] = evidence_pools
        self._stage_state["selected_evidence"] = selected_evidence
        self._stage_state["evidence_selection_trace"] = evidence_selection_trace
        return dataset

    def _postprocess_retrieval(self, dataset):
        self._stage_state["retrieval_postprocess"] = None
        self._stage_state["retrieval_postprocess_type"] = None
        self._stage_state["retrieval_postprocess_label"] = None

        if self.refiner is None:
            return dataset

        input_prompt_flag = self.refiner.input_prompt_flag
        if "llmlingua" in self.refiner.name and input_prompt_flag:
            input_prompts = [
                self.prompt_template.get_string(question=q, retrieval_result=r)
                for q, r in zip(dataset.question, dataset.retrieval_result)
            ]
            dataset.update_output("prompt", input_prompts)
            refined_prompts = self.refiner.batch_run(dataset)
            self._stage_state["retrieval_postprocess"] = refined_prompts
            self._stage_state["retrieval_postprocess_type"] = "prompt"
            self._stage_state["retrieval_postprocess_label"] = "Refining Prompts"
        else:
            refine_results = self.refiner.batch_run(dataset)
            dataset.update_output("refine_result", refine_results)
            self._stage_state["retrieval_postprocess"] = refine_results
            self._stage_state["retrieval_postprocess_type"] = "refine_result"
            self._stage_state["retrieval_postprocess_label"] = "Refining Documents"

        return dataset

    def _build_generation_input(self, dataset):
        stage_output = self._stage_state.get("retrieval_postprocess")
        stage_type = self._stage_state.get("retrieval_postprocess_type")

        if self.use_fid:
            print("Use FiD generation")
            input_prompts = []
            for item in dataset:
                q = item.question
                docs = item.retrieval_result
                input_prompts.append([q + " " + doc["contents"] for doc in docs])
        elif stage_type == "prompt":
            input_prompts = stage_output
        elif stage_type == "refine_result":
            input_prompts = [
                self.prompt_template.get_string(question=q, formatted_reference=r)
                for q, r in zip(dataset.question, stage_output)
            ]
        else:
            input_prompts = [
                self.prompt_template.get_string(question=q, retrieval_result=r)
                for q, r in zip(dataset.question, dataset.retrieval_result)
            ]

        dataset.update_output("prompt", input_prompts)
        self._stage_state["input_prompts"] = input_prompts
        return input_prompts

    def _pre_generate(self, dataset, input_prompts):
        """Hook for prompt augmentation before generation."""
        return input_prompts

    def _generate(self, dataset, input_prompts):
        pred_answer_list = self.generator.generate(input_prompts)
        dataset.update_output("pred", pred_answer_list)
        return pred_answer_list

    def _postprocess_generation(self, dataset):
        """Hook for answer post-processing after generation."""
        return dataset

    def _finalize(self, dataset, do_eval=True, pred_process_fun=None):
        return self.evaluate(dataset, do_eval=do_eval, pred_process_fun=pred_process_fun)

    def naive_run(self, dataset, do_eval=True, pred_process_fun=None):
        self._reset_stage_state()
        dataset = self._preprocess_query(dataset)

        input_prompts = [self.prompt_template.get_string(question=q) for q in dataset.question]
        dataset.update_output("prompt", input_prompts)

        input_prompts = self._pre_generate(dataset, input_prompts)
        dataset.update_output("prompt", input_prompts)

        self._generate(dataset, input_prompts)
        dataset = self._postprocess_generation(dataset)
        dataset = self._finalize(dataset, do_eval=do_eval, pred_process_fun=pred_process_fun)
        return dataset

    def run(self, dataset, do_eval=True, pred_process_fun=None):
        self._reset_stage_state()
        self._stage_state["enable_query_rewrite"] = True
        self._ensure_refiner()

        dataset = self._preprocess_query(dataset)
        dataset = self._retrieve(dataset)
        dataset = self._select_evidence(dataset)
        dataset = self._postprocess_retrieval(dataset)

        input_prompts = self._build_generation_input(dataset)
        input_prompts = self._pre_generate(dataset, input_prompts)
        dataset.update_output("prompt", input_prompts)

        self._release_refiner()
        self._generate(dataset, input_prompts)

        dataset = self._postprocess_generation(dataset)
        dataset = self._finalize(dataset, do_eval=do_eval, pred_process_fun=pred_process_fun)
        return dataset


class ConditionalPipeline(BasicPipeline):
    def __init__(self, config, prompt_template=None, retriever=None, generator=None):
        """
        inference stage:
            query -> judger -> sequential pipeline or naive generate
        """

        super().__init__(config, prompt_template)

        self.judger = get_judger(config)
        if generator is None:
            self.generator = get_generator(config)
        if retriever is None:
            self.retriever = get_retriever(config)
        self.generator = generator
        self.retriever = retriever

        self.sequential_pipeline = SequentialPipeline(
            config, prompt_template, retriever=self.retriever, generator=self.generator
        )

        self.zero_shot_templete = PromptTemplate(
            config=config,
            system_prompt="Answer the question based on your own knowledge. \
                            Only give me the answer and do not output any other words.",
            user_prompt="Question: {question}",
        )

    def run(self, dataset, do_eval=True, pred_process_fun=None):
        # judge_result: list of bool element, representing whether to use retrieval
        judge_result = self.judger.judge(dataset)
        dataset.update_output("judge_result", judge_result)

        # split dataset based on judge_result
        dataset_split = split_dataset(dataset, judge_result)
        pos_dataset, neg_dataset = dataset_split[True], dataset_split[False]

        pos_dataset = self.sequential_pipeline.run(pos_dataset, do_eval=False)
        self.sequential_pipeline.prompt_template = self.zero_shot_templete
        neg_dataset = self.sequential_pipeline.naive_run(neg_dataset, do_eval=False)

        # merge datasets into original format
        dataset = merge_dataset(dataset_split, judge_result)

        dataset = self.evaluate(dataset, do_eval=do_eval, pred_process_fun=pred_process_fun)

        return dataset


class AdaptivePipeline(BasicPipeline):
    def __init__(
        self,
        config,
        norag_template=None,
        single_hop_prompt_template=None,
        multi_hop_prompt_template=None,
        retriever = None,
        generator = None
    ):
        super().__init__(config)
        # load adaptive classifier as judger
        self.judger = get_judger(config)

        if generator is None:
            generator = get_generator(config)
        if retriever is None:
            retriever = get_retriever(config)

        self.generator = generator
        self.retriever = retriever

        # Load three pipeline for three types of query: naive/single-hop/multi-hop
        from flashrag.pipeline import IRCOTPipeline

        if norag_template is None:
            norag_templete = PromptTemplate(
                config=config,
                system_prompt="Answer the question based on your own knowledge. Only give me the answer and do not output any other words.",
                user_prompt="Question: {question}",
            )
        self.norag_pipeline = SequentialPipeline(
            config,
            prompt_template=norag_templete,
            retriever=retriever,
            generator=generator,
        )

        self.single_hop_pipeline = SequentialPipeline(
            config,
            prompt_template=single_hop_prompt_template,
            retriever=retriever,
            generator=generator,
        )

        self.multi_hop_pipeline = IRCOTPipeline(
            config, prompt_template=multi_hop_prompt_template, retriever=retriever, generator=generator, max_iter=5
        )

    def run(self, dataset, do_eval=True, pred_process_fun=None):
        # judge_result: choice result representing which pipeline to use(e.g. A, B, C)
        judge_result = self.judger.judge(dataset)
        dataset.update_output("judge_result", judge_result)

        # split dataset based on judge_result
        dataset_split = split_dataset(dataset, judge_result)
        for symbol, symbol_dataset in dataset_split.items():
            if symbol == "A":
                symbol_dataset = self.norag_pipeline.naive_run(symbol_dataset, do_eval=False)
            elif symbol == "B":
                symbol_dataset = self.single_hop_pipeline.run(symbol_dataset, do_eval=False)
            elif symbol == "C":
                symbol_dataset = self.multi_hop_pipeline.run(symbol_dataset, do_eval=False)
            else:
                assert False, "Unknown symbol!"

        # merge datasets into original format
        dataset = merge_dataset(dataset_split, judge_result)

        dataset = self.evaluate(dataset, do_eval=do_eval, pred_process_fun=pred_process_fun)

        return dataset
