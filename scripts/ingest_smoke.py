import asyncio

from sqlalchemy import func, select

from app.db import Chunk, Paper, SessionLocal
from app.ingest.arxiv_client import fetch_papers
from app.ingest.embedder import embed_texts
from app.ingest.repository import save_papers


async def main():
    results = fetch_papers("attention is all you need", max_results=3)
    print(f"fetched {len(results)} papers")
    for r in results:
        print(" -", r.title)

    async with SessionLocal() as session:
        inserted = await save_papers(session, results)
        print(f"inserted {inserted} papers")

        chunks = [
            Chunk(
                paper_id=r.entry_id.rsplit("/", 1)[-1],
                section="abstract",
                content=r.summary,
            )
            for r in results
        ]
        session.add_all(chunks)
        await session.commit()
        print(f"inserted {len(chunks)} chunks")

        pending = (
            await session.execute(select(Chunk).where(Chunk.embedding.is_(None)))
        ).scalars().all()
        print(f"embedding {len(pending)} chunks...")

        vectors = await embed_texts([c.content for c in pending])
        for chunk, vec in zip(pending, vectors):
            chunk.embedding = vec
        await session.commit()
        print(f"embedded {len(pending)} chunks")

        total_papers = await session.scalar(select(func.count()).select_from(Paper))
        total_chunks = await session.scalar(select(func.count()).select_from(Chunk))
        embedded = await session.scalar(
            select(func.count()).select_from(Chunk).where(Chunk.embedding.is_not(None))
        )
        print(f"papers in DB: {total_papers}")
        print(f"chunks in DB: {total_chunks} (embedded: {embedded})")


if __name__ == "__main__":
    asyncio.run(main())
