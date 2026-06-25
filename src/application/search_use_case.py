from __future__ import annotations

import psycopg

from src.infrastructure.database.age_repository import AgeRepository
from src.infrastructure.database.connection import get_postgres_connection


THEME_RELATION_HINTS = {
    "discursa_sobre",
    "narra_sobre",
    "aborda_tema",
    "trata_de",
    "fala_sobre",
    "comenta_sobre",
    "discute",
    "debate",
    "argumenta_sobre",
    "explica_sobre",
    "ocorre_em",
}


def _is_discurso_node(item: dict[str, str]) -> bool:
    source_label = item.get("source_label", "").upper()
    target_label = item.get("target_label", "").upper()
    return source_label == "DISCURSO" or target_label == "DISCURSO"


def _normalize_relation(value: str) -> str:
    relation = value.strip().strip('"').lower()
    return "_".join(relation.split())


def _is_theme_relation(relation: str) -> bool:
    normalized = _normalize_relation(relation)
    return (
        normalized in THEME_RELATION_HINTS
        or "_sobre" in normalized
        or "tema" in normalized
        or "assunto" in normalized
    )


class SearchUseCase:
    def __init__(self, conn: psycopg.Connection | None = None) -> None:
        self._conn = conn or get_postgres_connection()
        self._age_repo = AgeRepository(self._conn)
        self._age_repo.ensure_graph()

    def search(self, keyword: str, limit: int = 5) -> dict[str, list[dict[str, str]]]:
        graph_results = self._age_repo.search_graph(keyword, limit=max(10, limit * 2))
        theme_graph_results = [
            item
            for item in graph_results
            if _is_theme_relation(item.get("relation", "")) or _is_discurso_node(item)
        ]
        return {
            "graph": graph_results,
            "graph_theme": theme_graph_results,
        }
