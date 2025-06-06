import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from typing import Any

from datasets import load_dataset
from loguru import logger

from entrag.api.model import RAGLM
from entrag.config.evaluation_config import EvaluationConfig
from entrag.data_model.question_answer import EvaluationResult, InferenceResult, QuestionAnswerPair, Source
from entrag.evaluators import (
    answer_classification_llm_evaluator,
    answer_correctenss_llm_evaluator,
    ndcg_k_evaluator,
    precision_k_evaluator,
    recall_k_evaluator,
)
from entrag.prompts.default_prompts import DEFAULT_QA_SYSTEM_PROMPT, DEFAULT_QA_USER_PROMPT
from entrag.utils.prompt import get_formatted_chunks, get_formatted_external_chunks, get_query_time


def _load_dataset(config: EvaluationConfig) -> list[QuestionAnswerPair]:
    """
    Load QA dataset from either local file or Hugging Face Hub.

    Args:
        config: The evaluation configuration
    """
    dataset_config = config.tasks.question_answering

    if hasattr(dataset_config, "hf_dataset_id") and dataset_config.hf_dataset_id:
        logger.info(f"Loading dataset from Hugging Face: [{dataset_config.hf_dataset_id}]")
        try:
            hf_dataset = load_dataset(
                dataset_config.hf_dataset_id,
                split=getattr(dataset_config, "split", "train"),
                token=getattr(dataset_config, "hf_token", None),
            )
            dataset_items = [dict(item) for item in hf_dataset]
            logger.info(f"Loaded {len(dataset_items)} QA pairs from Hugging Face")

        except Exception as e:
            logger.error(f"Failed to load dataset from Hugging Face: {e}")
            raise ValueError("Failed to load dataset from Hugging Face. Please check the dataset ID.") from e

    elif dataset_config.dataset_path is not None:
        logger.info(f"Loading dataset from local file: [{dataset_config.dataset_path}]")
        with open(dataset_config.dataset_path, "r") as file:
            dataset_items = json.load(file)
        logger.info(f"Loaded {len(dataset_items)} QA pairs from local file")

    else:
        raise ValueError("Either hf_dataset_id or dataset_path must be specified in the configuration")

    return [QuestionAnswerPair(**item) for item in dataset_items]


def _compute_and_log_aggregated_metrics(output_file: str) -> None:
    """
    Compute aggregated metrics from results and add to `aggregated_metrics` in the output file.
    """
    if not os.path.exists(output_file):
        return

    with open(output_file, "r") as f:
        existing_results = json.load(f)

    all_metrics = {}
    classification_scores = []

    for question_data in existing_results["questions"].values():
        for eval_result in question_data["evaluation_results"]:
            evaluator_name = eval_result["evaluator_name"]
            score = eval_result["score"]

            if evaluator_name not in all_metrics:
                all_metrics[evaluator_name] = []
            all_metrics[evaluator_name].append(score)

            if evaluator_name == "answer_classification_llm":
                classification_scores.append(score)

    existing_results["aggregated_metrics"] = {}
    for evaluator_name, scores in all_metrics.items():
        existing_results["aggregated_metrics"][evaluator_name] = {
            "average": sum(scores) / len(scores) if scores else 0,
            "count": len(scores),
        }

    if classification_scores:
        total = len(classification_scores)
        perfect_count = classification_scores.count(1.0)
        acceptable_count = classification_scores.count(0.5)
        missing_count = classification_scores.count(0.0)
        incorrect_count = classification_scores.count(-1.0)

        existing_results["aggregated_metrics"]["classification_accuracy"] = {
            "average": ((perfect_count + acceptable_count) / total) * 100,
            "count": total,
        }
        existing_results["aggregated_metrics"]["classification_hallucination"] = {
            "average": (incorrect_count / total) * 100,
            "count": total,
        }
        existing_results["aggregated_metrics"]["classification_missing"] = {
            "average": (missing_count / total) * 100,
            "count": total,
        }
        existing_results["aggregated_metrics"]["classification_truthfulness"] = {
            "average": sum(classification_scores) / total,
            "count": total,
        }

    with open(output_file, "w") as f:
        json.dump(existing_results, f, indent=2)


def _log_evaluation_results(
    *,
    example: QuestionAnswerPair,
    inference_result: InferenceResult,
    eval_result: EvaluationResult,
    output_file: str,
    file_lock: Lock,
) -> None:
    """
    Log the evaluation results to a file.

    Args:
        example: The question-answer pair being evaluated
        inference_result: The model's inference result
        eval_result: The evaluation result
        output_file: Path to the JSON output file
        file_lock: Lock for safe file access
    """
    with file_lock:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        existing_results: dict[str, Any] = {}
        if os.path.exists(output_file):
            try:
                with open(output_file, "r") as f:
                    existing_results = json.load(f)
            except json.JSONDecodeError:
                logger.warning(f"Could not parse existing results file: [{output_file}]. Creating a new file.")
                existing_results = {"questions": {}}
        else:
            existing_results = {"questions": {}}

        if "questions" not in existing_results:
            existing_results["questions"] = {}

        question_id = str(example.id)
        if question_id not in existing_results["questions"]:
            existing_results["questions"][question_id] = {
                "question_id": question_id,
                "question": example.question,
                "ground_truth": example.reference_answer,
                "model_answer": inference_result.answer,
                "retrieved_sources": [
                    {"id": source.id, "filename": source.filename, "pages": source.pages}
                    for source in inference_result.sources
                ],
                "evaluation_results": [],
            }

        evaluator_exists = False
        for idx, result in enumerate(existing_results["questions"][question_id]["evaluation_results"]):
            if result["evaluator_name"] == eval_result.evaluator:
                existing_results["questions"][question_id]["evaluation_results"][idx] = {
                    "evaluator_name": eval_result.evaluator,
                    "score": eval_result.score,
                }
                evaluator_exists = True
                break

        if not evaluator_exists:
            existing_results["questions"][question_id]["evaluation_results"].append({
                "evaluator_name": eval_result.evaluator,
                "score": eval_result.score,
            })

        with open(output_file, "w") as f:
            json.dump(existing_results, f, indent=2)

        logger.debug(f"Updated evaluation results in {output_file}")


def _run_question_inference(
    model: RAGLM, example: QuestionAnswerPair, config: EvaluationConfig, output_file: str, file_lock: Lock
) -> list[EvaluationResult]:
    """
    Process a single question-answer pair and return evaluation results.

    Args:
        model: The model to evaluate
        example: The question-answer pair to process
        config: The evaluation configuration
        output_file: Path to the output file for logging results
        file_lock: Lock for safe file access

    Returns:
        List of evaluation results for this question
    """
    results: list[EvaluationResult] = []
    llm_evaluators = [answer_correctenss_llm_evaluator, answer_classification_llm_evaluator]

    try:
        retrieved_chunks, ext_chunks = model.retrieve(example.question, config.model_evaluation.retrieval_top_k)
        sources = [
            Source(id=str(idx + 1), filename=chunk.document_name, pages=[chunk.document_page])
            for idx, chunk in enumerate(retrieved_chunks)
        ]
        sources.extend(
            Source(id=str(f"api-{idx + 1}"), filename=ext_chunk.source, pages=[])
            for idx, ext_chunk in enumerate(ext_chunks)
        )

        answer = model.generate(
            system_prompt=DEFAULT_QA_SYSTEM_PROMPT,
            user_prompt=DEFAULT_QA_USER_PROMPT.format(
                query=example.question,
                query_time=get_query_time(),
                references=get_formatted_chunks(retrieved_chunks),
                additional_context=get_formatted_external_chunks(ext_chunks),
            ),
        )
        inference_result = InferenceResult(
            question_id=example.id,
            answer=answer,
            sources=sources,
        )

        # Run LLM evaluators
        for evaluator in llm_evaluators:
            eval_result = evaluator(example, inference_result)
            results.append(eval_result)
            _log_evaluation_results(
                example=example,
                inference_result=inference_result,
                eval_result=eval_result,
                output_file=output_file,
                file_lock=file_lock,
            )

        # Run additional evaluators
        additional_evals = [
            precision_k_evaluator(example, inference_result, k=config.model_evaluation.retrieval_top_k),
            recall_k_evaluator(example, inference_result, k=config.model_evaluation.retrieval_top_k),
            ndcg_k_evaluator(example, inference_result, k=config.model_evaluation.retrieval_top_k),
        ]

        for eval_result in additional_evals:
            results.append(eval_result)
            _log_evaluation_results(
                example=example,
                inference_result=inference_result,
                eval_result=eval_result,
                output_file=output_file,
                file_lock=file_lock,
            )

        logger.info(f"Completed processing question {example.id}")

    except Exception as e:
        logger.error(f"Error processing question {example.id}: {e}")

    return results


def evaluate_question_answering(model: RAGLM, config: EvaluationConfig, *, output_file: str) -> list[EvaluationResult]:
    """
    Evaluate the question answering task using the provided model and configuration.

    Args:
        model: The model to evaluate.
        config: The evaluation configuration.
        output_file: The path to the output file for logging results.
    """
    results: list[EvaluationResult] = []
    file_lock = Lock()

    dataset = _load_dataset(config)
    logger.info(f"Loaded [{len(dataset)}] QA pairs.")
    logger.debug(f"Running evaluation with [{config.model_evaluation.max_workers}] workers.")

    with ThreadPoolExecutor(max_workers=config.model_evaluation.max_workers) as executor:
        future_to_question = {
            executor.submit(_run_question_inference, model, example, config, output_file, file_lock): example
            for example in dataset
        }

        for future in as_completed(future_to_question):
            question = future_to_question[future]
            try:
                question_results = future.result()
                results.extend(question_results)
                logger.debug(f"Collected results for question {question.id}")
            except Exception as e:
                logger.error(f"Question {question.id} generated an exception: {e}")

    _compute_and_log_aggregated_metrics(output_file)
    return results
