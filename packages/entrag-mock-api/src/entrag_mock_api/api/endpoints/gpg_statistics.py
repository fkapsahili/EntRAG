import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query

from entrag_mock_api.api.deps import get_gpg_statistics_data
from entrag_mock_api.schema import GPGStatistic, GPGStatisticsResponse


router = APIRouter(tags=["GPG Statistics"])


@router.get("", response_model=GPGStatisticsResponse)
def get_gpg_statistics(
    employer_name: str = Query(..., min_length=3, max_length=100),
    data: pd.DataFrame = Depends(get_gpg_statistics_data),
):
    mask = data["EmployerName"].str.contains(employer_name, case=False, na=False)
    filtered_df = data[mask].head(10)  # Limit to 10 records

    if filtered_df.empty:
        raise HTTPException(
            status_code=404,
            detail=f"No GPG statistics found for employer name '{employer_name}'",
        )

    filtered_df = filtered_df.where(pd.notnull(filtered_df), None)
    records = filtered_df.to_dict(orient="records")
    results = [GPGStatistic(**row) for row in records]
    return GPGStatisticsResponse(statistics=results, total_count=len(results))
