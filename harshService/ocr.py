
"""
Medical OCR using Gemini 2.5 Flash.

Extracts text from medical report images while preserving
the original wording, values, units, tables and reading order.
"""

import mimetypes
from functools import lru_cache
from pathlib import Path

from google import genai
from google.genai import types

from config import settings


# Strict instructions for the OCR model
MEDICAL_OCR_SYSTEM_PROMPT = """
You are a specialized MEDICAL OCR ENGINE.

Your ONLY responsibility is to extract text from medical document images.

STRICT RULES:

1. Extract ALL visible text exactly as it appears in the image.
2. Do not summarize, interpret, correct or rewrite any content.
3. Preserve all medical terminology, spelling, numbers and symbols.
4. Preserve laboratory values, units, reference ranges and flags.
5. Preserve dates, patient details, headings and report sections.
6. Preserve the original reading order.
7. Represent tables using Markdown tables.
8. Keep section headings and paragraphs clearly separated.
9. Preserve decimal points, percentages, mathematical symbols
   and special characters.
10. If text is unreadable, write [ILLEGIBLE].
11. Never guess missing or unclear values.
12. Do not provide diagnoses, medical explanations or advice.
13. Do not add introductory text, conclusions or commentary.

IMPORTANT:
This is an OCR task, NOT a medical interpretation task.

Return ONLY the extracted document text.
"""


@lru_cache(maxsize=1)
def _get_client():
    """Create and reuse the Gemini API client."""
    return genai.Client(api_key=settings.GEMINI_API_KEY)


def extract_text(image_path: str) -> str:
    """
    Extract all visible text from a medical report image.

    Args:
        image_path: Path to the medical report image.

    Returns:
        Extracted text as a string.

    Raises:
        FileNotFoundError: If the image does not exist.
        ValueError: If the image format is unsupported.
        RuntimeError: If Gemini returns no usable text.
    """

    # 1. Validate image path
    path = Path(image_path)

    if not path.is_file():
        raise FileNotFoundError(
            f"Medical report image not found: {path}"
        )

    # 2. Detect image MIME type
    mime_type, _ = mimetypes.guess_type(str(path))

    supported_types = {
        "image/png",
        "image/jpeg",
        "image/webp",
        "image/heic",
        "image/heif",
    }

    if mime_type not in supported_types:
        raise ValueError(
            f"Unsupported image format: {mime_type}"
        )

    # 3. Read image as binary data
    image_bytes = path.read_bytes()

    if not image_bytes:
        raise ValueError("The supplied image is empty.")

    # 4. Create Gemini image part
    image_part = types.Part.from_bytes(
        data=image_bytes,
        mime_type=mime_type,
    )

    # 5. Call Gemini 2.5 Flash
    client = _get_client()

    response = client.models.generate_content(
        model=settings.OCR_MODEL_NAME,
        contents=[
            types.Part.from_text(
                text="Extract all visible text from this medical document."
            ),
            image_part,
        ],
        config=types.GenerateContentConfig(
            system_instruction=MEDICAL_OCR_SYSTEM_PROMPT,
            temperature=0,
        ),
    )

    # 6. Extract generated text
    extracted_text = response.text

    if not extracted_text or not extracted_text.strip():
        raise RuntimeError(
            "Gemini did not return any readable OCR text."
        )

    # 7. Return OCR result
    return extracted_text.strip()
