import asyncio

from sqlalchemy import func, select

from app.db import Chunk, Paper, SessionLocal
from app.ingest.arxiv_client import fetch_papers
from app.ingest.embedder import embed_texts
from app.ingest.repository import save_papers

QUERIES = [
    "cat:cs.CL",
    "cat:cs.AI",
    "cat:cs.LG",
]
PER_QUERY = 100
EMBED_BATCH = 100


async def ingest_papers() -> None:
    for query in QUERIES:
        print(f"fetching: {query} (max {PER_QUERY})")
        results = fetch_papers(query, max_results=PER_QUERY)
        print(f"  fetched {len(results)}")

        async with SessionLocal() as session:
            inserted = await save_papers(session, results)
            print(f"  inserted {inserted} new papers")


async def chunk_missing() -> int:
    async with SessionLocal() as session:
        papers = list(
            (
                await session.execute(select(Paper).where(~Paper.chunks.any()))
            ).scalars()
        )
        if not papers:
            print("no papers need chunking")
            return 0

        chunks = [
            Chunk(paper_id=p.id, section="abstract", content=p.abstract)
            for p in papers
        ]
        session.add_all(chunks)
        await session.commit()
        print(f"created {len(chunks)} chunks")
        return len(chunks)


async def embed_missing() -> None:
    async with SessionLocal() as session:
        pending = list(
            (
                await session.execute(
                    select(Chunk).where(Chunk.embedding.is_(None))
                )
            ).scalars()
        )
        if not pending:
            print("no chunks need embedding")
            return

        print(f"embedding {len(pending)} chunks (batch={EMBED_BATCH})...")
        for i in range(0, len(pending), EMBED_BATCH):
            batch = pending[i : i + EMBED_BATCH]
            vectors = await embed_texts([c.content for c in batch])
            for chunk, vec in zip(batch, vectors):
                chunk.embedding = vec
            await session.commit()
            print(f"  {i + len(batch)}/{len(pending)}")


async def summary() -> None:
    async with SessionLocal() as session:
        n_papers = await session.scalar(select(func.count()).select_from(Paper))
        n_chunks = await session.scalar(select(func.count()).select_from(Chunk))
        n_embedded = await session.scalar(
            select(func.count())
            .select_from(Chunk)
            .where(Chunk.embedding.is_not(None))
        )
    print(f"\n=== DB ===")
    print(f"papers   : {n_papers}")
    print(f"chunks   : {n_chunks}")
    print(f"embedded : {n_embedded}")


async def main() -> None:
    await ingest_papers()
    await chunk_missing()
    await embed_missing()
    await summary()


if __name__ == "__main__":
    asyncio.run(main())
