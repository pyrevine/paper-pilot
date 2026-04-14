from dataclasses import dataclass

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import Chunk


@dataclass
class FtsHit:
    chunk: Chunk
    score: float


async def fts_search(
    session: AsyncSession, query: str, top_k: int = 5
) -> list[FtsHit]:
    ts_query = func.websearch_to_tsquery("english", query)
    rank = func.ts_rank_cd(Chunk.content_tsv, ts_query).label("score")
    stmt = (
        select(Chunk, rank)
        .where(Chunk.content_tsv.op("@@")(ts_query))
        .order_by(rank.desc())
        .limit(top_k)
    )
    rows = (await session.execute(stmt)).all()
    return [FtsHit(chunk=c, score=float(s)) for c, s in rows]
