from entrag_mock_api.api.endpoints import finance
from fastapi import APIRouter


api_router = APIRouter()
api_router.include_router(finance.router, prefix="/finance", tags=["finance"])
