
import math
from google import genai
from google.genai import types
from config import settings


client = genai.Client(api_key=settings.GEMINI_API_KEY)


def embed_chunks(chunks: list[str]) -> list[list[float]]:
    """Embed report chunks using Google's hosted embedding API."""

    vectors = []

    for chunk in chunks:
        response = client.models.embed_content(
            model=settings.INGESTION_EMBEDDING_MODEL,
            contents=chunk,
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_DOCUMENT",
                output_dimensionality=768
            )
        )

        vector = response.embeddings[0].values

        norm = math.sqrt(sum(x * x for x in vector))

        if norm == 0:
            raise ValueError("Zero-length embedding vector")

        vectors.append([float(x / norm) for x in vector])

    return vectors
