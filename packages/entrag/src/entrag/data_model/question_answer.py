from typing import Literal

from pydantic import BaseModel, Field


class Source(BaseModel):
    """
    Model for a single source.
    """

    id: str = Field(description="The unique identifier for the source")
    filename: str = Field(description="The original filename of the source document, e.g. 'document.pdf'")
    pages: list[int] = Field(description="The page numbers in the source document that contain the answer")


class QuestionAnswerPair(BaseModel):
    """
    Model for a single question-answer pair.
    """

    id: str = Field(description="The unique identifier for the question")
    question_type: Literal[
        "simple",
        "simple_w_condition",
        "comparison",
        "aggregation",
        "multi_hop_reasoning",
        "factual_contradiction",
        "post_processing",
    ]
    question: str = Field(description="The question to answer")
    domain: Literal[
        "Finance",
        "Technical Documentation",
        "Environment",
        "Legal & Compliance",
        "Human Resources",
        "Marketing & Sales",
    ]
    dynamism: Literal["static", "dynamic"]
    reference_answer: str = Field(description="The reference answer to the question")
    sources: list[Source] = Field(description="The original source references that contain the answer")


class InferenceResult(BaseModel):
    """
    Model for the result of a RAG system's inference.
    """

    question_id: str = Field(description="The unique identifier for the question")
    answer: str = Field(description="The answer to the question")
    sources: list[Source] = Field(description="The used source references to generate the answer")


class EvaluationResult(BaseModel):
    """
    Model for the result of an evaluator on a question.
    """

    question_id: str = Field(description="The ID of the question this evaluation result refers to")
    evaluator: str = Field(description="The name of the evaluator")
    score: float = Field(description="The evaluation score for the question")


class LLMAnswerEvaluation(BaseModel):
    """
    Model used for structured outputs of an LLM-based evaluation.
    """

    score: int = Field(description="The score given by the evaluation for the answer")
    reasoning: str = Field(description="A short reasoning for the score given by the evaluation")
