import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from entrag_mock_api.api.deps import get_websites_data
from entrag_mock_api.api.utils import clean_html
from entrag_mock_api.schema import WebsiteResult, WebsiteSearchResponse


router = APIRouter(tags=["Search"])


@router.get("", response_model=WebsiteSearchResponse)
def search_websites(
    query: str = Query(..., min_length=3, max_length=100),
    data: list[WebsiteResult] = Depends(get_websites_data),
):
    docs = [f"{website.title} {clean_html(website.page_result)}" for website in data]

    # Compute relevance scores using TF-IDF weighted cosine similarity over both the website title and cleaned HTML.
    # We prioritize documents with semantically meaningful term overlap and filters out low-signal matches.
    vectorizer = TfidfVectorizer(stop_words="english")
    doc_vectors = vectorizer.fit_transform(docs)
    query_vector = vectorizer.transform([query])

    similarities = cosine_similarity(query_vector, doc_vectors).flatten()
    top_indices = np.argsort(similarities)[::-1][:10]
    top_results = [data[i] for i in top_indices if similarities[i] > 0.1]

    if not top_results:
        raise HTTPException(status_code=404, detail="No websites found matching the query")

    return {"results": top_results}
