from fastapi import Depends, FastAPI, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_db_session
from app.rag.hybrid import hybrid_search

app = FastAPI(title="Paper Pilot")


@app.get("/api/health")
async def health():
    return {"status": "ok"}


@app.get("/api/search")
async def search_endpoint(
    q: str = Query(..., min_length=1, description="검색 쿼리"),
    top_k: int = Query(5, ge=1, le=50),
    pool_k: int = Query(20, ge=1, le=100),
    session: AsyncSession = Depends(get_db_session),
):
    hits = await hybrid_search(session, q, top_k=top_k, pool_k=pool_k)
    return [
        {
            "paper_id": h.chunk.paper_id,
            "section": h.chunk.section,
            "rrf_score": h.score,
            "ranks": h.ranks,
            "content": h.chunk.content,
        }
        for h in hits
    ]
