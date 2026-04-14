from app.db import Chunk, Paper

def chunk_paper(paper: Paper) -> list[Chunk]:
    return [
        Chunk(
            paper_id=paper.id,
            section="abstract",
            content=paper.abstract,
        )
    ]