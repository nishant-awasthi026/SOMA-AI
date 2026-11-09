
from uuid import uuid4
from pathlib import Path

from qdrant_client import QdrantClient, models
from config import settings


def upsert_vectors(
    chunks: list[str],
    vectors: list[list[float]],
    *,
    user_id: str,
    source_image: str,
    report_date: str | None = None,
) -> list[str]:

    if not chunks or len(chunks) != len(vectors):
        raise ValueError("Chunks and vectors must be nonempty and match")

    dimensions = len(vectors[0])

    if any(len(vector) != dimensions for vector in vectors):
        raise ValueError("All vectors must have the same dimensions")

    client = QdrantClient(
        url=settings.QDRANT_URL,
        api_key=settings.QDRANT_API_KEY,
    )

    collection = settings.QDRANT_COLLECTION_NAME

    if not client.collection_exists(collection):
        client.create_collection(
            collection_name=collection,
            vectors_config=models.VectorParams(
                size=dimensions,
                distance=models.Distance.COSINE,
            ),
        )

    ids = [str(uuid4()) for _ in chunks]

    points = [
        models.PointStruct(
            id=ids[i],
            vector=vectors[i],
            payload={
                "text": chunks[i],
                "user_id": user_id,
                "source_image": Path(source_image).name,
                "chunk_index": i,
                "ReportDate": report_date,
            },
        )
        for i in range(len(chunks))
    ]

    client.upsert(
        collection_name=collection,
        points=points,
        wait=True,
    )

    return ids
