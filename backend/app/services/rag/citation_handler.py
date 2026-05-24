import re
from typing import Iterable, List

from app.models.document_models import Source


def answer_has_no_document_support(answer: str) -> bool:
    text = (answer or "").lower()
    no_answer_phrases = (
        "does not contain enough information",
        "could not find",
        "not found in the document",
        "not mentioned in the document",
        "not provided in the document",
        "no information",
        "insufficient information",
    )
    return any(phrase in text for phrase in no_answer_phrases)


def dedupe_sources(sources: Iterable[Source]) -> List[Source]:
    seen = set()
    unique: List[Source] = []
    for source in sources:
        key = (source.document_id, source.page, source.text[:80])
        if key not in seen:
            seen.add(key)
            unique.append(source)
    return unique


def pages_cited_in_answer(answer: str) -> List[int]:
    found: List[tuple[int, int]] = []
    seen = set()

    def add(position: int, page: int) -> None:
        if page > 0:
            found.append((position, page))

    for match in re.finditer(r"\bpages?\s+((?:\d+\s*(?:,|and|-|to)?\s*)+)", answer or "", flags=re.I):
        numbers = [int(value) for value in re.findall(r"\d+", match.group(1))]
        if len(numbers) >= 2 and ("-" in match.group(1) or re.search(r"\bto\b", match.group(1), flags=re.I)):
            start, end = numbers[:2]
            for page in range(start, end + 1):
                add(match.start(), page)
        else:
            for page in numbers:
                add(match.start(), page)

    for pattern in (r"\bp\.\s*(\d+)\b", r"\[\s*p(?:age)?\.?\s*(\d+)\s*\]"):
        for match in re.finditer(pattern, answer or "", flags=re.I):
            add(match.start(), int(match.group(1)))

    pages: List[int] = []
    for _, page in sorted(found, key=lambda item: item[0]):
        if page not in seen:
            seen.add(page)
            pages.append(page)
    return pages


def order_sources_for_answer(answer: str, sources: Iterable[Source]) -> List[Source]:
    if answer_has_no_document_support(answer):
        return []

    unique = dedupe_sources(sources)
    cited_pages = pages_cited_in_answer(answer)
    if not cited_pages:
        return unique

    page_rank = {page: index for index, page in enumerate(cited_pages)}
    cited = [source for source in unique if source.page in page_rank]
    cited.sort(key=lambda source: (page_rank[source.page], -source.score))
    best_by_page = {}
    for source in cited:
        if source.page not in best_by_page:
            best_by_page[source.page] = source
    return list(best_by_page.values())
