from arxiv import Result
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.dialects.postgresql import insert
from app.db.models import Paper, Chunk


def _result_to_row(r: Result) -> dict:
    return {
        "id": r.entry_id.rsplit("/", 1)[-1], 
        "title": r.title,
        "authors": [a.name for a in r.authors],
        "abstract": r.summary,
        "categories": r.categories,
        "published": r.published.date() if r.published else None,
        "updated": r.updated.date() if r.updated else None,
        "pdf_url": r.pdf_url
    }


async def save_papers(session: AsyncSession, results: list[Result]) -> int:
    if not results:
        return 0

    rows = [_result_to_row(r) for r in results]
    stmt = insert(Paper).values(rows).on_conflict_do_nothing(index_elements=["id"])
    result = await session.execute(stmt)
    await session.commit()

    return result.rowcount or 0


async def save_chunks(session: AsyncSession, chunks: list[Chunk]) -> int:
    if not chunks:
        return 0
    
    session.add_all(chunks)
    