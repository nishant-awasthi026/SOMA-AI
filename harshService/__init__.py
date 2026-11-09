
from pathlib import Path

from .ocr import extract_text
from .chunking import chunk_text
from .embedding import embed_chunks
from .upsert import upsert_vectors


def imgUpsert(
    image_path: str,
    *,
    user_id: str = "default_user",
    report_date: str | None = None,
) -> dict:
    """OCR -> chunking -> embeddings -> Qdrant."""

    text = extract_text(image_path)
    chunks = chunk_text(text)

    if not chunks:
        raise ValueError("No text chunks extracted")

    vectors = embed_chunks(chunks)

    ids = upsert_vectors(
        chunks,
        vectors,
        user_id=user_id,
        source_image=Path(image_path).name,
        report_date=report_date,
    )

    return {
        "user_id": user_id,
        "source_image": Path(image_path).name,
        "ReportDate": report_date,
        "chunks_uploaded": len(ids),
        "point_ids": ids,
    }
