from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import Chunk
from app.ingest.embedder import embed_texts


@dataclass
class SearchHit:
    chunk: Chunk
    distance: float


async def search(session: AsyncSession, query: str, top_k: int = 5) -> list[SearchHit]:
    [query_vec] = await embed_texts([query])
    distance = Chunk.embedding.cosine_distance(query_vec)
    stmt = (
        select(Chunk, distance.label("distance"))
        .where(Chunk.embedding.is_not(None))
        .order_by(distance)
        .limit(top_k)
    )
    rows = (await session.execute(stmt)).all()
    return [SearchHit(chunk=c, distance=float(d)) for c, d in rows]
