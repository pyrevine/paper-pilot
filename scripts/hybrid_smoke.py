import asyncio
import sys

from app.db import SessionLocal
from app.rag.hybrid import hybrid_search


async def main(query: str):
    async with SessionLocal() as session:
        hits = await hybrid_search(session, query, top_k=5, pool_k=20)

    if not hits:
        print(f"no hits for: {query}")
        return

    print(f"query: {query}\n")
    for i, hit in enumerate(hits, 1):
        ranks = "  ".join(f"{src}={r}" for src, r in hit.ranks.items())
        print(f"[{i}] rrf={hit.score:.4f}  ({ranks})  paper_id={hit.chunk.paper_id}")
        print(f"    {hit.chunk.content[:120]}...")
        print()


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "attention mechanism"
    asyncio.run(main(q))
