import os

from google import genai
from loguru import logger

from entrag.data_model.question_answer import InferenceResult, QuestionAnswerPair
from entrag.prompts.default_prompts import LLM_AS_JUDGE_CORRECTNESS_PROMPT


def answer_correctenss_llm_evaluator(example: QuestionAnswerPair, result: InferenceResult) -> int:
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
    return answer.text
