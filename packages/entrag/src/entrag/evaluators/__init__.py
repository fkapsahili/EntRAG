import os

import numpy as np
from loguru import logger
from openai import Client

from entrag.data_model.question_answer import (
    EvaluationResult,
    InferenceResult,
    LLMAnswerClassification,
    LLMAnswerEvaluation,
    QuestionAnswerPair,
)
from entrag.evaluators.utils import is_api_source, normalize_filename
from entrag.prompts.default_prompts import (
    LLM_AS_JUDGE_CLASSIFICATION_PROMPT,
    LLM_AS_JUDGE_CORRECTNESS_PROMPT,
    LLM_AS_JUDGE_SYSTEM_PROMPT,
)


def answer_correctenss_llm_evaluator(example: QuestionAnswerPair, result: InferenceResult) -> EvaluationResult:
    """
    Compute the quality of the answer based on the similarity between the reference answer and the generated answer.
    """
    prompt = LLM_AS_JUDGE_CORRECTNESS_PROMPT.format(
        query=example.question,
        reference_answer=example.reference_answer,
        answer=result.answer,
    )
    logger.debug(f"Answer correctness prompt: {prompt}")

    completion = Client(api_key=os.getenv("OPENAI_API_KEY")).beta.chat.completions.parse(
        model="gpt-4.1",
        messages=[{"role": "system", "content": LLM_AS_JUDGE_SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
        response_format=LLMAnswerEvaluation,
    )
    answer = completion.choices[0].message.parsed
    logger.debug(f"Answer correctness response: {answer.score} - {answer.reasoning}")
    return EvaluationResult(question_id=example.id, evaluator="answer_correctness_llm", score=float(answer.score))


def answer_classification_llm_evaluator(example: QuestionAnswerPair, result: InferenceResult) -> EvaluationResult:
    """
    Classify the answer into perfect/acceptable/missing/incorrect categories.
    """
    prompt = LLM_AS_JUDGE_CLASSIFICATION_PROMPT.format(
        query=example.question,
        reference_answer=example.reference_answer,
        answer=result.answer,
    )

    completion = Client(api_key=os.getenv("OPENAI_API_KEY")).beta.chat.completions.parse(
        model="gpt-4.1",
        messages=[{"role": "system", "content": LLM_AS_JUDGE_SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
        response_format=LLMAnswerClassification,
    )

    classification = completion.choices[0].message.parsed
    logger.debug(f"Answer classification: {classification.category} - {classification.reasoning}")

    score_map = {"perfect": 1.0, "acceptable": 0.5, "missing": 0.0, "incorrect": -1.0}
    return EvaluationResult(
        question_id=example.id, evaluator="answer_classification_llm", score=score_map[classification.category]
    )


def precision_k_evaluator(example: QuestionAnswerPair, result: InferenceResult, *, k: int) -> EvaluationResult:
    """
    Compute how frequently the correct chunks appear within the top-k retrieved chunks.
    """

    correct_pages_by_file = {normalize_filename(source.filename): set(source.pages) for source in example.sources}
    matches = set()
    for src in result.sources[:k]:
        filename = normalize_filename(src.filename)
        if filename in correct_pages_by_file:
            if is_api_source(filename) or not correct_pages_by_file[filename]:
                matches.add((filename, 0))  # We use 0 as a dummy page number
                continue

            for page in src.pages:
                if page in correct_pages_by_file[filename] and (filename, page) not in matches:
                    matches.add((filename, page))
                    break

    precision_at_k = len(matches) / k if k > 0 else 0.0
    return EvaluationResult(question_id=example.id, evaluator="precision_at_k", score=precision_at_k)


def recall_k_evaluator(example: QuestionAnswerPair, result: InferenceResult, *, k: int) -> EvaluationResult:
    """
    Compute how frequently the correct chunks appear within the top-k retrieved chunks.
    """

    correct_pages_by_file = {normalize_filename(source.filename): set(source.pages) for source in example.sources}
    matches = set()
    for src in result.sources[:k]:
        filename = normalize_filename(src.filename)
        if filename in correct_pages_by_file:
            if is_api_source(filename) or not correct_pages_by_file[filename]:
                matches.add((filename, 0))  # We use 0 as a dummy page number
                continue

            for page in src.pages:
                if page in correct_pages_by_file[filename]:
                    matches.add((filename, page))

    total_expected = 0
    for filename, pages in correct_pages_by_file.items():
        if is_api_source(filename) or not pages:
            total_expected += 1
        else:
            total_expected += len(pages)

    recall_at_k = len(matches) / total_expected if total_expected > 0 and k > 0 else 0.0
    return EvaluationResult(question_id=example.id, evaluator="recall_at_k", score=recall_at_k)


def ndcg_k_evaluator(example: QuestionAnswerPair, result: InferenceResult, *, k: int) -> EvaluationResult:
    """
    Compute the Normalized Discounted Cumulative Gain (NDCG) at k.
    """
    correct_pages_by_file = {normalize_filename(source.filename): set(source.pages) for source in example.sources}
    dcg = 0.0
    for i, src in enumerate(result.sources[:k]):
        filename = normalize_filename(src.filename)
        if filename in correct_pages_by_file:
            if is_api_source(filename) or not correct_pages_by_file[filename]:
                dcg += 1 / np.log2(i + 2)
                continue

            if any(page in correct_pages_by_file[filename] for page in src.pages):
                dcg += 1 / np.log2(i + 2)

    total_ideal_sources = sum(
        1 for filename, pages in correct_pages_by_file.items() if filename.startswith("api-") or pages
    )
    ideal_dcg = sum(1 / np.log2(i + 2) for i in range(min(total_ideal_sources, k)))
    score = dcg / ideal_dcg if ideal_dcg > 0 else 0.0
    return EvaluationResult(question_id=example.id, evaluator="ndcg_at_k", score=score)
