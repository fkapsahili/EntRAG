from entrag.api.model import Document, RAGLM


class OpenAIRAG(RAGLM):
    def __init__(self) -> None:
        """
        OpenAI completions model.
        """
        super().__init__()

    def retrieve(self, query: str, top_k: int = 5) -> list[Document]:
        raise NotImplementedError

    def generate(
        self,
        context: str,
        retrieved_docs: list[Document],
        generation_kwargs: dict | None = None,
    ) -> str:
        raise NotImplementedError

    def evaluate(self, queries: list[str]) -> list[str]:
        raise NotImplementedError
