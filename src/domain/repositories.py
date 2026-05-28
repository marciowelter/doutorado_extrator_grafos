from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from src.domain.models import KnowledgeGraphExtraction


@dataclass(frozen=True)
class VectorSearchResult:
    content: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class GraphSearchResult:
    source: str
    relation: str
    target: str


class ChunkRepository(ABC):
    @abstractmethod
    def fetch_chunks(self, limit: int) -> list[str]:
        raise NotImplementedError


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


class VectorRepository(ABC):
    @abstractmethod
    def ensure_store(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def upsert_chunk(self, chunk_text: str, metadata: dict[str, str] | None = None) -> None:
        raise NotImplementedError

    @abstractmethod
    def similarity_search(self, query: str, limit: int = 5) -> list[VectorSearchResult]:
        raise NotImplementedError


class KnowledgeExtractor(ABC):
    @abstractmethod
    def extract(self, text: str) -> KnowledgeGraphExtraction:
        raise NotImplementedError
