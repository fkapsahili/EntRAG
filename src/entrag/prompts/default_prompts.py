# ruff: noqa: E501
import textwrap


SIMPLE_QA_PROMPT = textwrap.dedent("""
You are given a Question, References and the time when it was asked in the Central European Time Zone,
referred to as "Query Time". The query time is formatted as "yyyy-mm-dd hh:mm:ss". Your task is to
answer the question based on the references provided.

Please follow these guidelines when formatting your answer:
1. If the questoin contains a false premise or assumption, answer "invalid question".
2. If you are uncertain or don't know the answer, answer "I don't know".

### Question
{query}

### Query Time
{query_time}

### References
{references}

### Answer
""")
