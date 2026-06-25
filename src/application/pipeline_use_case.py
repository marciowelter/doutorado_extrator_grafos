from __future__ import annotations

import time
import psycopg

from src.domain.discurso_context import DiscursoContext
from src.domain.models import Entity, KnowledgeGraphExtraction
from src.domain.repositories import GraphRepository, KnowledgeExtractor
from src.infrastructure.database.age_repository import AgeRepository
from src.infrastructure.database.connection import get_postgres_connection
from src.infrastructure.llm.llamaindex_client import LlamaIndexKnowledgeExtractor


_LAST_PIPELINE_TIMING: dict[str, float] = {
    "extract_seconds": 0.0,
    "save_seconds": 0.0,
    "total_seconds": 0.0,
}


def _set_last_pipeline_timing(payload: dict[str, float]) -> None:
    global _LAST_PIPELINE_TIMING
    _LAST_PIPELINE_TIMING = payload


def get_last_pipeline_timing() -> dict[str, float]:
    return _LAST_PIPELINE_TIMING


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

    def process_text(
        self,
        text: str,
        additional_themes: list[str] | None = None,
        discurso_context: DiscursoContext | None = None,
    ) -> KnowledgeGraphExtraction:
        started_at = time.perf_counter()
        extract_started_at = time.perf_counter()
        extraction = self.extract_text(
            text,
            additional_themes=additional_themes,
            discurso_context=discurso_context,
        )
        extract_seconds = round(time.perf_counter() - extract_started_at, 4)

        save_started_at = time.perf_counter()
        self.save_extraction(extraction)
        save_seconds = round(time.perf_counter() - save_started_at, 4)

        _set_last_pipeline_timing(
            {
                "extract_seconds": extract_seconds,
                "save_seconds": save_seconds,
                "total_seconds": round(time.perf_counter() - started_at, 4),
            }
        )
        return extraction

    def extract_text(
        self,
        text: str,
        additional_themes: list[str] | None = None,
        discurso_context: DiscursoContext | None = None,
        cached_themes: list[Entity] | None = None,
    ) -> KnowledgeGraphExtraction:
        return self._extractor.extract(
            text,
            additional_themes=additional_themes,
            discurso_context=discurso_context,
            cached_themes=cached_themes,
        )

    def save_extraction(self, extraction: KnowledgeGraphExtraction) -> None:
        started_at = time.perf_counter()
        self._refresh_connection_if_needed()
        self._age_repo.save_extraction(extraction)
        save_seconds = round(time.perf_counter() - started_at, 4)
        _set_last_pipeline_timing(
            {
                "extract_seconds": get_last_pipeline_timing().get("extract_seconds", 0.0),
                "save_seconds": save_seconds,
                "total_seconds": round(
                    get_last_pipeline_timing().get("extract_seconds", 0.0) + save_seconds,
                    4,
                ),
            }
        )

    def normalize_and_unify_graph(self) -> dict[str, int]:
        self._refresh_connection_if_needed()
        return self._age_repo.normalize_and_unify_graph_entities()
