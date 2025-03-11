import json

from entrag.config.evaluation_config import EvaluationConfig


def evaluate_question_answering(config: EvaluationConfig):
    prompts = []

    with open(config.tasks.question_answering.dataset_path, "r") as file:
        dataset = json.load(file)
        for question in dataset:
            user_prompt = ""  # TODO
            prompt = user_prompt.format(question=question["question"])
