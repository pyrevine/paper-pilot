import uuid
from dataclasses import dataclass

from app.db import Chunk


@dataclass
class FusedHit:
    chunk: Chunk
    score: float
    ranks: dict[str, int]


def reciprocal_rank_fusion(
    named_lists: dict[str, list[Chunk]],
    k: int = 60,
) -> list[FusedHit]:
    scores: dict[uuid.UUID, float] = {}
    ranks_map: dict[uuid.UUID, dict[str, int]] = {}
    chunk_map: dict[uuid.UUID, Chunk] = {}

    for source, chunks in named_lists.items():
        for rank, chunk in enumerate(chunks, start=1):
            scores[chunk.id] = scores.get(chunk.id, 0.0) + 1.0 / (k + rank)
            ranks_map.setdefault(chunk.id, {})[source] = rank
            chunk_map[chunk.id] = chunk

    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return [
        FusedHit(chunk=chunk_map[cid], score=score, ranks=ranks_map[cid])
        for cid, score in ordered
    ]
