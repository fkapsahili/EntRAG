import math

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query
from rapidfuzz import fuzz, process

from entrag_mock_api.api.deps import get_gpg_statistics_data
from entrag_mock_api.schema import GPGStatistic, GPGStatisticsResponse


router = APIRouter(tags=["GPG Statistics"])


def find_employer_matches(employer_name: str, data: pd.DataFrame, threshold: int = 70) -> pd.DataFrame:
    """
    Find potential employer matches using different matching strategies.
    """
    employer_name_lower = employer_name.lower().strip()

    # Strategy 1: Exact substring match
    exact_mask = data["EmployerName"].str.contains(employer_name, case=False, na=False, regex=False)
    exact_matches = data[exact_mask]

    if not exact_matches.empty:
        return exact_matches.head(10)

    # Strategy 2: Fuzzy matching for close matches
    employer_names = data["EmployerName"].dropna().tolist()

    fuzzy_matches = process.extract(
        employer_name,
        employer_names,
        scorer=fuzz.partial_ratio,
        limit=20,
    )

    # Filter by threshold and get indices
    good_matches = [match for match in fuzzy_matches if match[1] >= threshold]

    if good_matches:
        matched_names = [match[0] for match in good_matches]
        fuzzy_mask = data["EmployerName"].isin(matched_names)
        return data[fuzzy_mask].head(10)

    # Strategy 3: Word-based matching
    search_words = set(employer_name_lower.split())
    if len(search_words) > 0:
        word_scores = []
        for idx, name in enumerate(data["EmployerName"].dropna()):
            name_words = set(name.lower().split())
            # Calculate word overlap score
            overlap = len(search_words.intersection(name_words))
            if overlap > 0:
                score = (overlap / len(search_words)) * 100
                if score >= 50:  #
                    word_scores.append((idx, score))

        if word_scores:
            # Sort by score and take top matches
            word_scores.sort(key=lambda x: x[1], reverse=True)
            indices = [score[0] for score in word_scores[:10]]
            return data.iloc[indices]

    return pd.DataFrame()


@router.get("", response_model=GPGStatisticsResponse)
def get_gpg_statistics(
    employer_name: str = Query(..., min_length=3, max_length=100),
    data: pd.DataFrame = Depends(get_gpg_statistics_data),
):
    filtered_df = find_employer_matches(employer_name, data)

    if filtered_df.empty:
        raise HTTPException(
            status_code=404,
            detail=f"No GPG statistics found for employer name '{employer_name}'. Try a different spelling or shorter name.",
        )

    records = filtered_df.replace({float("nan"): None}).to_dict(orient="records")
    cleaned_records = []

    for row in records:
        cleaned_row = {}
        for k, v in row.items():
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                cleaned_row[k] = None
            elif v is None or (isinstance(v, str) and v.strip() == ""):
                cleaned_row[k] = "" if k in ["CompanyNumber", "EmployerName"] else None
            else:
                cleaned_row[k] = v
        cleaned_records.append(cleaned_row)

    results = [GPGStatistic(**row) for row in cleaned_records]
    return GPGStatisticsResponse(statistics=results, total_count=len(results), year="2024/2025")
