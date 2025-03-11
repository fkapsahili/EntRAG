from typing import Literal

from pydantic import BaseModel, Field


class QuestionAnswerPair(BaseModel):
    """
    Model for a single question-answer pair.
    """

    question_id: str = Field(description="The unique identifier for the question")
    question_type = Literal["simple", "simple_w_condition", "comparison", "aggregation"]
    question: str = Field(description="The question to answer")
    domain = Literal["Finance"]
    dynamism: Literal["static", "dynamic"]
    answer: str = Field(description="The answer to the question")
    reference_answer: str = Field(description="The reference answer to the question")
