from fastapi import APIRouter, Depends, HTTPException

from entrag_mock_api.api.deps import get_filings_data
from entrag_mock_api.schema import FilingsByTypeResponse, FilingSearchResponse


router = APIRouter(tags=["Filings"])


@router.get("/type/{filing_type}", response_model=FilingsByTypeResponse)
def get_filings(filing_type: str, company: str = None, data=Depends(get_filings_data)):
    # Normalize filing type
    data = get_filings_data()
    filing_type = filing_type.replace("%20", " ").strip()

    if filing_type not in data:
        raise HTTPException(
            status_code=404, detail=f"Filing type '{filing_type}' not found. Available types: {list(data.keys())}"
        )

    filings = data[filing_type]

    if company:
        filings = [f for f in filings if company.lower() in f["company"].lower()]
        if not filings:
            raise HTTPException(
                status_code=404, detail=f"No {filing_type} filings found for company matching '{company}'"
            )

    return {"filings": filings}


@router.get("/search", response_model=FilingSearchResponse)
def search_filings(filing_type: str = None, company: str = None, data=Depends(get_filings_data)):
    results = []

    if filing_type and filing_type not in data:
        raise HTTPException(status_code=400, detail=f"Invalid filing type: {filing_type}")

    for f_type, filings in data.items():
        if filing_type and f_type != filing_type:
            continue

        for filing in filings:
            if company and company.lower() not in filing["company"].lower():
                continue

            results.append(filing)

    return {"results": results}
