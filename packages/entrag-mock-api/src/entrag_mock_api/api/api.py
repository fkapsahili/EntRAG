from fastapi import APIRouter

from entrag_mock_api.api.endpoints import filings, finance, gpg_statistics, search


api_router = APIRouter()
api_router.include_router(finance.router, prefix="/finance", tags=["finance"])
api_router.include_router(filings.router, prefix="/filings", tags=["filings"])
api_router.include_router(gpg_statistics.router, prefix="/gpg-statistics", tags=["GPG Statistics"])
api_router.include_router(search.router, prefix="/search", tags=["Search"])
