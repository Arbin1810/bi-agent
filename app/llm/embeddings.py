from openai import OpenAI
from config.settings import settings

kwargs = {"api_key": settings.openai_api_key}
if settings.openai_base_url:
    kwargs["base_url"] = settings.openai_base_url
_client = OpenAI(**kwargs)


def embed(text: str) -> list[float]:
    resp = _client.embeddings.create(
        model=settings.openai_embedding_model,
        input=text,
    )
    return resp.data[0].embedding


def embed_batch(texts: list[str]) -> list[list[float]]:
    resp = _client.embeddings.create(
        model=settings.openai_embedding_model,
        input=texts,
    )
    return [d.embedding for d in resp.data]