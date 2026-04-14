from openai import AsyncOpenAI

_client = AsyncOpenAI()
_MODEL = "text-embedding-3-small"


async def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    resp = await _client.embeddings.create(model=_MODEL, input=texts)
    return [d.embedding for d in resp.data]
