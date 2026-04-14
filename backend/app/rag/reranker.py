import asyncio
from dataclasses import dataclass
from functools import lru_cache

from sentence_transformers import CrossEncoder

from app.db import Chunk
from app.rag.rrf import FusedHit

_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@dataclass
class RerankedHit:
    chunk: Chunk
    rerank_score: float
    rrf_score: float
    ranks: dict[str, int]


@lru_cache(maxsize=1)
def _get_model() -> CrossEncoder:
    return CrossEncoder(_MODEL_NAME)


def _score_sync(query: str, chunks: list[Chunk]) -> list[float]:
    if not chunks:
        return []
    model = _get_model()
    pairs = [(query, c.content) for c in chunks]
    return [float(s) for s in model.predict(pairs)]


async def rerank_fused(
    query: str, fused: list[FusedHit]
) -> list[RerankedHit]:
    if not fused:
        return []
    scores = await asyncio.to_thread(
        _score_sync, query, [f.chunk for f in fused]
    )
    indexed = list(zip(fused, scores))
    indexed.sort(key=lambda x: x[1], reverse=True)
    return [
        RerankedHit(
            chunk=f.chunk,
            rerank_score=s,
            rrf_score=f.score,
            ranks=f.ranks,
        )
        for f, s in indexed
    ]
