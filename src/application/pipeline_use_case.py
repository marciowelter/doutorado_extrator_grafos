from __future__ import annotations

import psycopg

from src.domain.models import KnowledgeGraphExtraction
from src.domain.repositories import GraphRepository, KnowledgeExtractor
from src.infrastructure.database.age_repository import AgeRepository
from src.infrastructure.database.connection import get_postgres_connection
from src.infrastructure.llm.llamaindex_client import LlamaIndexKnowledgeExtractor


class PipelineUseCase:
    def __init__(
        self,
        conn: psycopg.Connection | None = None,
        extractor: KnowledgeExtractor | None = None,
        graph_repo: GraphRepository | None = None,
    ) -> None:
        self._conn = conn or get_postgres_connection()
        self._extractor = extractor or LlamaIndexKnowledgeExtractor()
        self._age_repo = graph_repo or AgeRepository(self._conn)

    def bootstrap(self) -> None:
        self._age_repo.ensure_graph()

    def _refresh_connection_if_needed(self) -> None:
        try:
            self._conn.execute("SELECT 1")
        except Exception:
            self._conn = get_postgres_connection()
            self._age_repo._conn = self._conn  # type: ignore[attr-defined]

    def process_text(self, text: str) -> KnowledgeGraphExtraction:
        extraction = self._extractor.extract(text)
        self._refresh_connection_if_needed()
        self._age_repo.save_extraction(extraction)
        return extraction
