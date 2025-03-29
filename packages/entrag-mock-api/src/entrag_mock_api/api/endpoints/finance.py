from fastapi import APIRouter


router = APIRouter()


@router.get("/company/{ticker}")
async def get_company_data(ticker: str):
    """
    Get financial data for a specific company.
    """
    return {"ticker": ticker, "data": "Company financial data here."}
