import arxiv

search = arxiv.Search(
    query="attention is all you need",
    max_results=3
)

for result in arxiv.Client().results(search=search):
    print(result.title)
    print(result.authors)
    print(result.summary[:200])
    print("-----")