from typing import Iterable, List

from app.models.document_models import Source


def dedupe_sources(sources: Iterable[Source]) -> List[Source]:
    seen = set()
    unique: List[Source] = []
    for source in sources:
        key = (source.document_id, source.page, source.text[:80])
        if key not in seen:
            seen.add(key)
            unique.append(source)
    return unique
