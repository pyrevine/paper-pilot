from fastapi import Depends, FastAPI, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_db_session
from app.rag.hybrid import hybrid_search
from app.rag.reranker import rerank_fused

app = FastAPI(title="Paper Pilot")


@app.get("/api/health")
async def health():
    return {"status": "ok"}


@app.get("/api/search")
async def search_endpoint(
    q: str = Query(..., min_length=1, description="검색 쿼리"),
    top_k: int = Query(5, ge=1, le=50),
    pool_k: int = Query(20, ge=1, le=100),
    rerank: bool = Query(True, description="Cross-encoder 재순위"),
    session: AsyncSession = Depends(get_db_session),
):
    fused = await hybrid_search(session, q, top_k=pool_k, pool_k=pool_k)

    if not rerank:
        return [
            {
                "paper_id": f.chunk.paper_id,
                "section": f.chunk.section,
                "rrf_score": f.score,
                "ranks": f.ranks,
                "content": f.chunk.content,
            }
            for f in fused[:top_k]
        ]

    reranked = await rerank_fused(q, fused)
    return [
        {
            "paper_id": r.chunk.paper_id,
            "section": r.chunk.section,
            "rerank_score": r.rerank_score,
            "rrf_score": r.rrf_score,
            "ranks": r.ranks,
            "content": r.chunk.content,
        }
        for r in reranked[:top_k]
    ]
