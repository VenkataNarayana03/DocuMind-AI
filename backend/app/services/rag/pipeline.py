import asyncio
import json
import math
import re
import shutil
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from fastapi import UploadFile

from app.core.settings import get_settings
from app.models.document_models import ChatMessage, Document, Flashcard, PreviewPage, QuizQuestion, Source
from app.services.documents.document_loader import extract_document_text
from app.services.llm.groq_client import GroqClient
from app.services.llm.prompts import FLASHCARD_PROMPT, QUIZ_PROMPT, SUMMARY_PROMPT
from app.services.rag.chunking import TextChunk, chunk_pages
from app.services.rag.citation_handler import order_sources_for_answer
from app.services.rag.embeddings import EmbeddingService
from app.services.rag.pinecone_store import VectorStore
from app.utils.helpers import safe_filename


class RAGPipeline:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.embeddings = EmbeddingService()
        self.store = VectorStore(self.embeddings.dimension)
        self.llm = GroqClient()
        self.session_id = str(uuid.uuid4())
        self.document_counter = 0
        self.documents: Dict[str, Document] = {}
        self.chunks: Dict[str, List[TextChunk]] = {}
        self.pages: Dict[str, List[PreviewPage]] = {}
        self.files: Dict[str, Path] = {}
        self.upload_dir = Path(__file__).resolve().parents[3] / "uploads" / self.session_id
        self.upload_dir.mkdir(parents=True, exist_ok=True)

    async def ingest(self, files: List[UploadFile]) -> List[Document]:
        payloads = [(file.filename or "document", await file.read()) for file in files]
        return await asyncio.to_thread(self.ingest_bytes, payloads)

    def ingest_bytes(self, files: List[Tuple[str, bytes]]) -> List[Document]:
        docs: List[Document] = []
        for filename, content in files:
            document_id = self._next_document_id()
            suffix = Path(filename).suffix or ".pdf"
            stored_path = self.upload_dir / f"{document_id}-{safe_filename(filename)}"
            stored_path.write_bytes(content)
            try:
                pages = extract_document_text(stored_path)
            except Exception:
                stored_path.unlink(missing_ok=True)
                raise

            chunks = chunk_pages(document_id, filename, pages, self.settings.chunk_size, self.settings.chunk_overlap)
            vectors = self.embeddings.embed(chunk.text for chunk in chunks)
            self.store.upsert(
                {
                    "id": chunk.id,
                    "values": vector,
                    "metadata": {
                        "document_id": chunk.document_id,
                        "filename": chunk.filename,
                        "page": chunk.page,
                        "text": chunk.text,
                        "session_id": self.session_id,
                    },
                }
                for chunk, vector in zip(chunks, vectors)
            )
            doc = Document(
                id=document_id,
                filename=filename,
                pages=len(pages),
                chunks=len(chunks),
                scanned_pages=sum(1 for page in pages if page.used_ocr),
                file_type=suffix.lstrip(".").lower(),
            )
            self.documents[document_id] = doc
            self.chunks[document_id] = chunks
            self.pages[document_id] = [PreviewPage(page=page.page, text=page.text) for page in pages]
            self.files[document_id] = stored_path
            docs.append(doc)
        return docs

    def _next_document_id(self) -> str:
        self.document_counter += 1
        return f"doc{self.document_counter}"

    def ask(self, question: str, document_ids: Optional[List[str]], history: List[ChatMessage]) -> tuple[str, List[Source]]:
        sources = self.retrieve(question, document_ids)
        answer = self.llm.answer(question, sources, history)
        return answer, order_sources_for_answer(answer, sources)

    def retrieve(self, query: str, document_ids: Optional[List[str]] = None, top_k: Optional[int] = None) -> List[Source]:
        limit = top_k or self.settings.top_k
        candidate_limit = max(limit * 4, 12)
        terms = self._query_terms(query)
        vector = self.embeddings.embed([query])[0]
        vector_sources = self.store.query(vector, candidate_limit, document_ids, self.session_id)
        lexical_sources = self._lexical_search(query, document_ids, candidate_limit, terms)
        page_sources = self._page_context_sources(lexical_sources, document_ids, limit)
        if self._is_direct_section_lookup(terms, lexical_sources):
            return self._merge_sources(lexical_sources, page_sources, limit)

        ranked_sources = self._merge_sources(vector_sources, lexical_sources, candidate_limit)
        nearby_sources = self._nearby_sources(ranked_sources[:limit], document_ids)
        return self._merge_sources(ranked_sources, page_sources + nearby_sources, limit)

    def _lexical_search(self, query: str, document_ids: Optional[List[str]], limit: int, terms: Optional[List[str]] = None) -> List[Source]:
        terms = terms or self._query_terms(query)
        if not terms:
            return []

        ids = document_ids or list(self.documents)
        chunks = [chunk for doc_id in ids for chunk in self.chunks.get(doc_id, [])]
        doc_count = max(len(chunks), 1)
        document_frequency: Dict[str, int] = {}
        for chunk in chunks:
            chunk_terms = set(self._tokens(chunk.text))
            for term in terms:
                if term in chunk_terms:
                    document_frequency[term] = document_frequency.get(term, 0) + 1

        query_phrase = " ".join(terms)
        results: List[Source] = []
        for chunk in chunks:
            tokens = self._tokens(chunk.text)
            token_count = max(len(tokens), 1)
            term_counts: Dict[str, int] = {}
            for token in tokens:
                if token in terms:
                    term_counts[token] = term_counts.get(token, 0) + 1
            if not term_counts:
                continue

            matched_terms = len(term_counts)
            coverage = matched_terms / len(terms)
            score = 0.0
            for term, count in term_counts.items():
                idf = math.log((doc_count + 1) / (document_frequency.get(term, 0) + 1)) + 1.0
                tf = count / token_count
                score += idf * (1.0 + math.log1p(count)) * (1.0 + tf)

            text = chunk.text.lower()
            if query_phrase and query_phrase in text:
                score += 2.0
            if re.search(rf"(^|\n|\.)\s*(?:\d+[\.\)]\s*)?{' '.join(re.escape(term) for term in terms[:3])}", text):
                score += 0.75

            results.append(
                Source(
                    document_id=chunk.document_id,
                    filename=chunk.filename,
                    page=chunk.page,
                    text=chunk.text,
                    score=1.0 + score * coverage,
                )
            )

        results.sort(key=lambda source: source.score, reverse=True)
        return results[:limit]

    def _page_context_sources(self, sources: List[Source], document_ids: Optional[List[str]], limit: int) -> List[Source]:
        allowed = set(document_ids or self.documents.keys())
        page_keys: List[tuple[str, int]] = []
        seen_pages: Set[tuple[str, int]] = set()
        for source in sources:
            key = (source.document_id, source.page)
            if source.document_id in allowed and key not in seen_pages:
                page_keys.append(key)
                seen_pages.add(key)
            if len(page_keys) >= 2:
                break

        results: List[Source] = []
        for page_rank, (doc_id, page) in enumerate(page_keys):
            base_score = max((source.score for source in sources if source.document_id == doc_id and source.page == page), default=1.0)
            page_chunks = [chunk for chunk in self.chunks.get(doc_id, []) if chunk.page == page]
            for chunk_index, chunk in enumerate(page_chunks):
                results.append(
                    Source(
                        document_id=chunk.document_id,
                        filename=chunk.filename,
                        page=chunk.page,
                        text=chunk.text,
                        score=base_score - (page_rank * 0.25) - (chunk_index * 0.03),
                    )
                )
                if len(results) >= limit:
                    return results
        return results

    @staticmethod
    def _is_direct_section_lookup(terms: List[str], lexical_sources: List[Source]) -> bool:
        if not terms or not lexical_sources:
            return False
        if len(terms) <= 2:
            return True
        top_text = lexical_sources[0].text.lower()[:220]
        return all(re.search(rf"\b{re.escape(term)}\b", top_text) for term in terms[:2])

    def _nearby_sources(self, sources: List[Source], document_ids: Optional[List[str]]) -> List[Source]:
        allowed = set(document_ids or self.documents.keys())
        seen = {(source.document_id, source.text) for source in sources}
        nearby: List[Source] = []
        for source in sources:
            if source.document_id not in allowed:
                continue
            chunks = self.chunks.get(source.document_id, [])
            for index, chunk in enumerate(chunks):
                if chunk.page != source.page or chunk.text != source.text:
                    continue
                for neighbor_index in (index - 1, index + 1):
                    if 0 <= neighbor_index < len(chunks):
                        neighbor = chunks[neighbor_index]
                        key = (neighbor.document_id, neighbor.text)
                        if key in seen or neighbor.page != source.page:
                            continue
                        seen.add(key)
                        nearby.append(
                            Source(
                                document_id=neighbor.document_id,
                                filename=neighbor.filename,
                                page=neighbor.page,
                                text=neighbor.text,
                                score=source.score * 0.85,
                            )
                        )
                break
        return nearby

    @staticmethod
    def _tokens(text: str) -> List[str]:
        tokens = re.findall(r"\b[a-z0-9][a-z0-9-]{2,}\b", text.lower())
        expanded: List[str] = []
        for token in tokens:
            expanded.append(token)
            stem = RAGPipeline._simple_stem(token)
            if stem != token:
                expanded.append(stem)
        return expanded

    @staticmethod
    def _simple_stem(term: str) -> str:
        for suffix in ("ing", "ed", "es", "s"):
            if len(term) > len(suffix) + 3 and term.endswith(suffix):
                return term[: -len(suffix)]
        return term

    @staticmethod
    def _query_terms(query: str) -> List[str]:
        stop = {
            "about",
            "after",
            "all",
            "also",
            "and",
            "any",
            "answer",
            "are",
            "ask",
            "before",
            "can",
            "contain",
            "contains",
            "define",
            "does",
            "for",
            "from",
            "give",
            "has",
            "have",
            "here",
            "how",
            "into",
            "its",
            "more",
            "not",
            "show",
            "tell",
            "that",
            "the",
            "there",
            "this",
            "was",
            "what",
            "when",
            "where",
            "which",
            "with",
        }
        terms: List[str] = []
        seen: Set[str] = set()
        for token in re.findall(r"\b[a-z0-9][a-z0-9-]{2,}\b", query.lower()):
            for term in (token, RAGPipeline._simple_stem(token)):
                if term not in stop and term not in seen:
                    terms.append(term)
                    seen.add(term)
        return terms

    @staticmethod
    def _merge_sources(vector_sources: List[Source], lexical_sources: List[Source], limit: int) -> List[Source]:
        merged: Dict[tuple[str, int, str], Source] = {}
        for source in vector_sources + lexical_sources:
            key = (source.document_id, source.page, source.text)
            existing = merged.get(key)
            if not existing or source.score > existing.score:
                merged[key] = source
        return sorted(merged.values(), key=lambda source: source.score, reverse=True)[:limit]

    def all_sources(self, document_ids: Optional[List[str]] = None, limit: int = 10) -> List[Source]:
        ids = document_ids or list(self.documents)
        sources: List[Source] = []
        for doc_id in ids:
            sources.extend(
                Source(document_id=chunk.document_id, filename=chunk.filename, page=chunk.page, text=chunk.text, score=1.0)
                for chunk in self.chunks.get(doc_id, [])[:limit]
            )
        return sources[:limit]

    def summarize(self, document_ids: Optional[List[str]]) -> tuple[str, List[Source]]:
        sources = self.all_sources(document_ids, 12)
        summary = self.llm.complete(SUMMARY_PROMPT, sources, 900)
        return summary, sources[:5]

    def topics(self, document_ids: Optional[List[str]]) -> List[str]:
        text = " ".join(source.text for source in self.all_sources(document_ids, 20)).lower()
        words = re.findall(r"\b[a-z][a-z-]{4,}\b", text)
        stop = {"about", "which", "their", "there", "these", "those", "would", "could", "should", "where", "after", "before", "between"}
        counts: Dict[str, int] = {}
        for word in words:
            if word not in stop:
                counts[word] = counts.get(word, 0) + 1
        return [word.title() for word, _ in sorted(counts.items(), key=lambda item: item[1], reverse=True)[:12]]

    def quiz(self, document_ids: Optional[List[str]], count: int) -> List[QuizQuestion]:
        sources = self.all_sources(document_ids, 12)
        prompt = f"{QUIZ_PROMPT} Return JSON only as an array with question, options, answer. Make {count} questions."
        raw = self.llm.complete(prompt, sources, 1200)
        parsed = self._parse_json(raw, [])
        if isinstance(parsed, list) and parsed:
            return [QuizQuestion(**item) for item in parsed[:count] if {"question", "options", "answer"} <= set(item)]
        return [
            QuizQuestion(
                question=f"What is a key idea on page {source.page} of {source.filename}?",
                options=[source.text[:90], "Not discussed in the document", "A formatting instruction", "A file upload error"],
                answer=source.text[:90],
            )
            for source in sources[:count]
        ]

    def flashcards(self, document_ids: Optional[List[str]]) -> List[Flashcard]:
        sources = self.all_sources(document_ids, 12)
        prompt = f"{FLASHCARD_PROMPT} Return JSON only as an array with front and back. Make 8 cards."
        raw = self.llm.complete(prompt, sources, 1000)
        parsed = self._parse_json(raw, [])
        if isinstance(parsed, list) and parsed:
            return [Flashcard(**item) for item in parsed[:8] if {"front", "back"} <= set(item)]
        return [Flashcard(front=f"{source.filename} page {source.page}", back=source.text[:260]) for source in sources[:8]]

    def delete(self, document_id: str) -> None:
        vector_ids = [chunk.id for chunk in self.chunks.get(document_id, [])]
        self.documents.pop(document_id, None)
        self.chunks.pop(document_id, None)
        self.pages.pop(document_id, None)
        path = self.files.pop(document_id, None)
        if path:
            path.unlink(missing_ok=True)
        self.store.delete_document(document_id, vector_ids)

    def cleanup_session(self) -> None:
        vector_ids = [chunk.id for chunks in self.chunks.values() for chunk in chunks]
        for document_id in list(self.documents):
            self.documents.pop(document_id, None)
            self.chunks.pop(document_id, None)
            self.pages.pop(document_id, None)
            self.files.pop(document_id, None)
        self.store.delete_session(self.session_id, vector_ids)
        shutil.rmtree(self.upload_dir, ignore_errors=True)

    @staticmethod
    def _parse_json(raw: str, fallback):
        try:
            match = re.search(r"\[.*\]", raw, flags=re.S)
            return json.loads(match.group(0) if match else raw)
        except Exception:
            return fallback


pipeline = RAGPipeline()
