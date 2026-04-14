from sqlalchemy.ext.asyncio import AsyncSession

from app.rag.fts import fts_search
from app.rag.rrf import FusedHit, reciprocal_rank_fusion
from app.rag.search import search as vector_search


async def hybrid_search(
    session: AsyncSession,
    query: str,
    top_k: int = 5,
    pool_k: int = 20,
) -> list[FusedHit]:
    vec_hits = await vector_search(session, query, top_k=pool_k)
    fts_hits = await fts_search(session, query, top_k=pool_k)

    fused = reciprocal_rank_fusion(
        {
            "vector": [h.chunk for h in vec_hits],
            "fts": [h.chunk for h in fts_hits],
        }
    )
    return fused[:top_k]
