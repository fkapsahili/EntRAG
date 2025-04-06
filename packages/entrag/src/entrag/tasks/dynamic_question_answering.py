import json

from entrag.api.model import RAGLM
from entrag.config.evaluation_config import EvaluationConfig
from entrag.data_model.question_answer import EvaluationResult, QuestionAnswerPair


def evaluate_dynamic_question_answering(model: RAGLM, config: EvaluationConfig):
    evaluators = []
    resluts: list[EvaluationResult] = []

    with open(config.tasks.dynamic_question_answering.dataset_path, "r") as file:
        dataset_ = json.load(file)
        # Filter out non-dynamic QA pairs
        dataset = [QuestionAnswerPair(**item) for item in dataset_ if item["dynamism"] == "dynamic"]

        for example in dataset:
            model.retrieve(example.question)
