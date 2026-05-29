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

    def reconnect(self) -> None:
        self._conn = get_postgres_connection()
        self._age_repo = AgeRepository(self._conn)
        self._age_repo.ensure_graph()

    def _refresh_connection_if_needed(self) -> None:
        try:
            self._conn.execute("SELECT 1")
        except Exception:
            self.reconnect()

    def process_text(self, text: str) -> KnowledgeGraphExtraction:
        extraction = self._extractor.extract(text)
        self._refresh_connection_if_needed()
        self._age_repo.save_extraction(extraction)
        return extraction

    def normalize_and_unify_graph(self) -> dict[str, int]:
        self._refresh_connection_if_needed()
        return self._age_repo.normalize_and_unify_graph_entities()
