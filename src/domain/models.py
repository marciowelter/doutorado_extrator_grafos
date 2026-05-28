from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class Entity(BaseModel):
    name: str = Field(min_length=1)
    label: str = Field(default="ENTIDADE", min_length=1)
    properties: dict[str, Any] = Field(default_factory=dict)


class Relationship(BaseModel):
    source: str = Field(min_length=1)
    target: str = Field(min_length=1)
    relation: str = Field(min_length=1)
    properties: dict[str, Any] = Field(default_factory=dict)


class KnowledgeGraphExtraction(BaseModel):
    entities: list[Entity] = Field(default_factory=list)
    relationships: list[Relationship] = Field(default_factory=list)
