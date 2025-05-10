from fastapi import APIRouter

from entrag_mock_api.api.endpoints import filings, finance


api_router = APIRouter()
api_router.include_router(finance.router, prefix="/finance", tags=["finance"])
api_router.include_router(filings.router, prefix="/filings", tags=["filings"])
