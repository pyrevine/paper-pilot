import arxiv

def fetch_papers(query: str, max_results: int = 5) -> list[arxiv.Result]:
    results = []
    search = arxiv.Search(query=query, max_results=max_results)
    for result in arxiv.Client().results(search=search):
        results.append(result)
    
    return result

