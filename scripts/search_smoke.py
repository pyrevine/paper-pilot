import asyncio
import sys

from app.db import SessionLocal
from app.rag.search import search


async def main(query: str):
    async with SessionLocal() as session:
        hits = await search(session, query, top_k=5)

    if not hits:
        print("no hits (chunks 테이블이 비었거나 embedding이 없음)")
        return

    print(f"query: {query}\n")
    for i, hit in enumerate(hits, 1):
        print(f"[{i}] distance={hit.distance:.4f}  paper_id={hit.chunk.paper_id}")
        print(f"    {hit.chunk.content[:120]}...")
        print()


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "attention mechanism for long sequences"
    asyncio.run(main(q))
