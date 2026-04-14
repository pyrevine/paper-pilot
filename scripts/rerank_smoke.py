import asyncio
import sys
import time

from app.db import SessionLocal
from app.rag.hybrid import hybrid_search
from app.rag.reranker import rerank_fused


async def main(query: str):
    async with SessionLocal() as session:
        t0 = time.perf_counter()
        fused = await hybrid_search(session, query, top_k=20, pool_k=20)
        t1 = time.perf_counter()

        print(f"[hybrid] {len(fused)} fused hits in {(t1 - t0) * 1000:.0f}ms\n")
        for i, f in enumerate(fused[:5], 1):
            print(f"  [{i}] rrf={f.score:.4f}  {f.ranks}  {f.chunk.paper_id}")
        print()

        t2 = time.perf_counter()
        reranked = await rerank_fused(query, fused)
        t3 = time.perf_counter()

        print(f"[rerank] done in {(t3 - t2) * 1000:.0f}ms\n")
        for i, r in enumerate(reranked[:5], 1):
            print(
                f"  [{i}] rerank={r.rerank_score:+.2f}  rrf={r.rrf_score:.4f}  "
                f"{r.ranks}  {r.chunk.paper_id}"
            )
            print(f"      {r.chunk.content[:100]}...")


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "attention mechanism"
    asyncio.run(main(q))
