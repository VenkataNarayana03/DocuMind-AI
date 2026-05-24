from dataclasses import dataclass
from typing import Iterable, List


@dataclass
class TextChunk:
    id: str
    document_id: str
    filename: str
    page: int
    text: str


def chunk_pages(document_id: str, filename: str, pages: Iterable, size: int, overlap: int) -> List[TextChunk]:
    chunks: List[TextChunk] = []
    step = max(size - overlap, 1)
    chunk_index = 0
    for page in pages:
        text = " ".join(page.text.split())
        if not text:
            continue
        start = 0
        while start < len(text):
            body = text[start : start + size].strip()
            if body:
                chunks.append(
                    TextChunk(
                        id=f"{document_id}-{chunk_index}",
                        document_id=document_id,
                        filename=filename,
                        page=page.page,
                        text=body,
                    )
                )
                chunk_index += 1
            start += step
    return chunks
