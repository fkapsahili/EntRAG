import json

from entrag.api.model import RAGLM
from entrag.config.evaluation_config import EvaluationConfig
from entrag.data_model.document import Chunk
from entrag.data_model.question_answer import InferenceResult, QuestionAnswerPair
from entrag.evaluators import answer_correctenss_llm_evaluator, ndcg_k_evaluator, recall_k_evaluator
from entrag.prompts.default_prompts import SIMPLE_QA_PROMPT
from entrag.utils.prompt import get_query_time


def evaluate_question_answering(model: RAGLM, config: EvaluationConfig):
    prompts = []  # TODO: Parallelize the model inference
    evaluators = [answer_correctenss_llm_evaluator]
    results = []

    with open(config.tasks.question_answering.dataset_path, "r") as file:
        dataset_ = json.load(file)
        dataset = [QuestionAnswerPair(**item) for item in dataset_]

        for example in dataset:
            retrieved_chunks = model.retrieve(example.question)
            prompt = SIMPLE_QA_PROMPT.format(
                query=example.question,
                query_time=get_query_time(),
                references=_get_formatted_chunks(retrieved_chunks),
            )
            answer = model.generate(prompt)
            result = InferenceResult(
                question_id=example.id, answer=answer, sources=[chunk.document_id for chunk in retrieved_chunks]
            )
            for evaluator in evaluators:
                eval_result = evaluator(example, result)
                results.append(eval_result)

            results.extend([
                recall_k_evaluator(example, result, k=5),
                ndcg_k_evaluator(example, result, k=5),
            ])


def _get_formatted_chunks(chunks: list[Chunk]) -> str:
    """
    Format the chunks in an LLM-readable format.
    """
    return "\n".join([f"  - {chunk.chunk_text}" for chunk in chunks])
