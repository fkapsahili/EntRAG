from typing import Literal

from pydantic import BaseModel, Field


class QuestionAnswerPair(BaseModel):
    """
    Model for a single question-answer pair.
    """

    id: str = Field(description="The unique identifier for the question")
    question_type: Literal["simple", "simple_w_condition", "comparison", "aggregation"]
    question: str = Field(description="The question to answer")
    domain: Literal["Finance"]
    dynamism: Literal["static", "dynamic"]
    reference_answer: str = Field(description="The reference answer to the question")


class InferenceResult(BaseModel):
    """
    Model for the result of a RAG system's inference.
    """

    question_id: str = Field(description="The unique identifier for the question")
    answer: str = Field(description="The answer to the question")
    sources: list[str] = Field(description="The sources / document IDs used to generate the answer")
