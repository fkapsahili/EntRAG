import os

import numpy as np
from google import genai
from loguru import logger

from entrag.data_model.question_answer import EvaluationResult, InferenceResult, QuestionAnswerPair
from entrag.prompts.default_prompts import LLM_AS_JUDGE_CORRECTNESS_PROMPT


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
    answer = genai.Client(api_key=os.getenv("GEMINI_API_KEY")).models.generate_content(
        model="gemini-2.0-flash", contents=[prompt]
    )
    logger.debug(f"Answer correctness response: {answer}")
    answer_float = float(answer.text.strip())  # TODO: Use structured outputs
    return EvaluationResult(evaluator="answer_correctness_llm", score=answer_float)


def recall_k_evaluator(example: QuestionAnswerPair, result: InferenceResult, *, k: int) -> EvaluationResult:
    """
    Compute how frequently the correct chunks appear within the top-k retrieved chunks.
    """
    correct_chunks = set(example.sources).intersection(result.sources[:k])
    recall_at_k = len(correct_chunks) / len(example.sources) if example.sources else 0.0
    return EvaluationResult(evaluator="recall_at_k", score=recall_at_k)


def ndcg_k_evaluator(example: QuestionAnswerPair, result: InferenceResult, *, k: int) -> EvaluationResult:
    """
    Compute the Normalized Discounted Cumulative Gain (NDCG) at k.
    """
    ideal_sources = set(example.sources)
    dcg = 0.0
    for i, src in enumerate(result.sources[:k]):
        if src in ideal_sources:
            dcg += 1 / np.log2(i + 2)

    ideal_dcg = sum(1 / np.log2(i + 2) for i in range(min(len(ideal_sources), k)))
    score = dcg / ideal_dcg if ideal_dcg > 0 else 0.0
    return EvaluationResult(evaluator="ndcg_at_k", score=score)
