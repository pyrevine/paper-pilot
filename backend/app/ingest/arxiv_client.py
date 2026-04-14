import arxiv

def fetch_papers(query: str, max_results: int = 5) -> list[arxiv.Result]:
    client = arxiv.Client(
        page_size=max_results,
        delay_seconds=3,
        num_retries=3,
    )
    search = arxiv.Search(query=query, max_results=max_results)

    return list(client.results(search=search))
    

