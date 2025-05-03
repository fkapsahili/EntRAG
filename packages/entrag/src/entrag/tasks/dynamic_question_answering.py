import json

from loguru import logger

from entrag.api.model import RAGLM
from entrag.config.evaluation_config import EvaluationConfig
from entrag.data_model.question_answer import EvaluationResult, InferenceResult, QuestionAnswerPair, Source
from entrag.evaluators import answer_correctenss_llm_evaluator
from entrag.prompts.default_prompts import DYNAMIC_QA_PROMPT
from entrag.utils.prompt import get_formatted_chunks, get_query_time


def evaluate_dynamic_question_answering(model: RAGLM, config: EvaluationConfig) -> list[EvaluationResult]:
    """
    Evaluate the dynamic question answering task using the provided model and configuration.
    """
    evaluators = [answer_correctenss_llm_evaluator]
    resluts: list[EvaluationResult] = []

    with open(config.tasks.dynamic_question_answering.dataset_path, "r") as file:
        dataset_ = json.load(file)
        # Filter out non-dynamic QA pairs
        dataset = [QuestionAnswerPair(**item) for item in dataset_ if item["dynamism"] == "dynamic"]
        logger.info(f"Loaded {len(dataset)} dynamic QA pairs.")

        for example in dataset:
            retrieved_chunks, additional_context = model.retrieve(example.question)

            prompt = DYNAMIC_QA_PROMPT.format(
                query=example.question,
                query_time=get_query_time(),
                references=get_formatted_chunks(retrieved_chunks),
                additional_context="\n".join(additional_context),
            )
            answer = model.generate(prompt)
            result = InferenceResult(
                question_id=example.id,
                answer=answer,
                sources=[
                    Source(id=str(idx + 1), filename=chunk.document_name, pages=[chunk.document_page])
                    for idx, chunk in enumerate(retrieved_chunks)
                ],
            )
            for evaluator in evaluators:
                eval_result = evaluator(example, result)
                resluts.append(eval_result)

    return resluts
