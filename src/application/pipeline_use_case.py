from __future__ import annotations

import psycopg

from config.settings import settings
from src.domain.models import KnowledgeGraphExtraction
from src.domain.repositories import ChunkRepository, GraphRepository, KnowledgeExtractor, VectorRepository
from src.infrastructure.database.age_repository import AgeRepository
from src.infrastructure.database.chunk_repository import PostgresChunkRepository
from src.infrastructure.database.connection import get_postgres_connection
from src.infrastructure.database.vector_repository import VectorRepository as PgVectorRepository
from src.infrastructure.llm.llamaindex_client import LlamaIndexKnowledgeExtractor


class PipelineUseCase:
    def __init__(
        self,
        conn: psycopg.Connection | None = None,
        extractor: KnowledgeExtractor | None = None,
        chunk_repo: ChunkRepository | None = None,
        graph_repo: GraphRepository | None = None,
        vector_repo: VectorRepository | None = None,
    ) -> None:
        self._conn = conn or get_postgres_connection()
        self._extractor = extractor or LlamaIndexKnowledgeExtractor()
        self._chunk_repo = chunk_repo or PostgresChunkRepository(self._conn)
        self._age_repo = graph_repo or AgeRepository(self._conn)
        self._vector_repo = vector_repo or PgVectorRepository(self._conn)

    def bootstrap(self) -> None:
        self._age_repo.ensure_graph()
        self._vector_repo.ensure_store()

    def _refresh_connection_if_needed(self) -> None:
        try:
            self._conn.execute("SELECT 1")
        except Exception:
            self._conn = get_postgres_connection()
            self._age_repo._conn = self._conn  # type: ignore[attr-defined]
            self._vector_repo._conn = self._conn  # type: ignore[attr-defined]

    def process_text(self, text: str) -> KnowledgeGraphExtraction:
        extraction = self._extractor.extract(text)
        self._refresh_connection_if_needed()
        self._age_repo.save_extraction(extraction)
        self._vector_repo.upsert_chunk(text, metadata={"source": "llamaindex_native"})
        return extraction

    def process_source_chunks(self, limit: int | None = None) -> int:
        chunks = self._chunk_repo.fetch_chunks(limit=limit or settings.chunk_limit)
        processed = 0
        for chunk in chunks:
            if not chunk.strip():
                continue
            self.process_text(chunk)
            processed += 1
        return processed
