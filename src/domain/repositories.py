from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from src.domain.discurso_context import DiscursoContext
from src.domain.models import Entity, KnowledgeGraphExtraction


@dataclass(frozen=True)
class GraphSearchResult:
    source: str
    relation: str
    target: str


class GraphRepository(ABC):
    @abstractmethod
    def ensure_graph(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def save_extraction(self, extraction: KnowledgeGraphExtraction) -> None:
        raise NotImplementedError

    @abstractmethod
    def search_graph(self, keyword: str, limit: int = 20) -> list[GraphSearchResult]:
        raise NotImplementedError

    @abstractmethod
    def normalize_and_unify_graph_entities(self) -> dict[str, int]:
        raise NotImplementedError


class KnowledgeExtractor(ABC):
    @abstractmethod
    def extract(
        self,
        text: str,
        additional_themes: list[str] | None = None,
        discurso_context: DiscursoContext | None = None,
        cached_themes: list[Entity] | None = None,
    ) -> KnowledgeGraphExtraction:
        raise NotImplementedError
