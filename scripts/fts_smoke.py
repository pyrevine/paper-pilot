import asyncio
import sys

from app.db import SessionLocal
from app.rag.fts import fts_search


async def main(query: str):
    async with SessionLocal() as session:
        hits = await fts_search(session, query, top_k=5)

    if not hits:
        print(f"no hits for: {query}")
        return

    print(f"query: {query}\n")
    for i, hit in enumerate(hits, 1):
        print(f"[{i}] score={hit.score:.4f}  paper_id={hit.chunk.paper_id}")
        print(f"    {hit.chunk.content[:120]}...")
        print()


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "attention mechanism"
    asyncio.run(main(q))
