import json

from entrag.api.model import RAGLM
from entrag.config.evaluation_config import EvaluationConfig
from entrag.data_model.question_answer import EvaluationResult, InferenceResult, QuestionAnswerPair, Source
from entrag.prompts.default_prompts import DYNAMIC_QA_PROMPT
from entrag.utils.prompt import get_formatted_chunks, get_query_time


def evaluate_dynamic_question_answering(model: RAGLM, config: EvaluationConfig):
    evaluators = []
    resluts: list[EvaluationResult] = []

    with open(config.tasks.dynamic_question_answering.dataset_path, "r") as file:
        dataset_ = json.load(file)
        # Filter out non-dynamic QA pairs
        dataset = [QuestionAnswerPair(**item) for item in dataset_ if item["dynamism"] == "dynamic"]

        for example in dataset:
            retrieved_chunks, additional_context = model.retrieve(example.question)

            prompt = DYNAMIC_QA_PROMPT.format(
                query=example.question,
                query_time=get_query_time(),
                references=get_formatted_chunks(retrieved_chunks),
                additional_context="\n".join(additional_context),
            )
            answer = model.generate(prompt)
            print("--" * 20)
            print("prompt", prompt)
            print()
            print(f"Answer: {answer}")
            print("--" * 20)
            print()
            # result = EvaluationResult(
            #     question_id=example.id,
            #     answer=answer,
            #     sources=[
            #         Source(id=str(idx + 1), filename=chunk.document_name, pages=[chunk.document_page])
            #         for idx, chunk in enumerate(retrieved_chunks)
            #     ],
            # )
