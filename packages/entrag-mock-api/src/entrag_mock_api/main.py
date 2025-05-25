import uvicorn
from fastapi import FastAPI

from entrag_mock_api.api.api import api_router


app = FastAPI(title="EntRAG Mock API")

app.include_router(api_router, prefix="/api")


def start():
    uvicorn.run("entrag_mock_api.main:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    start()
