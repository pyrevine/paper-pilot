
async def save_papers(session: AsyncSession, papers: list[Paper]) -> None:
    # 논문 디비에 저장하기