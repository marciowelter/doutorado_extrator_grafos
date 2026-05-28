from __future__ import annotations

import json
import re
import unicodedata
from typing import Any

from llama_index.core import Settings
from llama_index.core.prompts import PromptTemplate
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from pydantic import BaseModel, Field

from config.settings import settings
from src.domain.models import Entity, KnowledgeGraphExtraction, Relationship
from src.domain.repositories import KnowledgeExtractor


MAX_DEBUG_CHARS = 400
_LAST_EXTRACTION_DEBUG: dict[str, Any] = {
    "attempts": [],
    "fallback_used": False,
}


ENTITY_EXTRACTION_PROMPT = PromptTemplate(
    """
Voce e um especialista em extracao de entidades em portugues brasileiro.
Extraia entidades relevantes do texto e devolva SOMENTE JSON valido seguindo o schema.

Regras obrigatorias:
1) Nao invente entidades.
2) Use o nome da entidade exatamente como aparece no texto, quando possivel.
2.1) Nao generalize pessoas com cargo. Se aparecer "Dep. Marquito", mantenha exatamente "Dep. Marquito".
3) Para categoria, use valores como: PESSOA, ORGANIZACAO, LOCAL, DATA, VALOR, TEMA, EVENTO.
4) contexto deve ser curto (uma frase) explicando por que a entidade e relevante.
5) Se nao houver entidades, retorne lista vazia.

Texto:
{text}
""".strip()
)


RELATION_EXTRACTION_PROMPT = PromptTemplate(
    """
Voce e um especialista em extracao de relacoes em portugues brasileiro.
Com base no texto e na lista de entidades previamente extraidas, identifique possiveis relacoes.
Devolva SOMENTE JSON valido seguindo o schema.

Regras obrigatorias:
1) Use somente entidades existentes na lista fornecida.
2) origem e destino devem ser os nomes exatos das entidades fornecidas.
3) relacao deve ser curta, em portugues brasileiro, usando snake_case quando possivel.
4) evidencia deve ser um trecho curto do texto que sustenta a relacao.
5) Se nao houver relacoes, retorne lista vazia.

Texto:
{text}

Entidades previamente extraidas (JSON):
{entities_json}
""".strip()
)


class StructuredEntity(BaseModel):
    nome: str = Field(min_length=1, description="Nome exato da entidade")
    categoria: str = Field(min_length=1, description="Categoria da entidade")
    contexto: str = Field(default="", description="Resumo curto da relevancia")


class StructuredEntityExtraction(BaseModel):
    entidades: list[StructuredEntity] = Field(default_factory=list)


class StructuredRelation(BaseModel):
    origem: str = Field(min_length=1, description="Entidade de origem")
    destino: str = Field(min_length=1, description="Entidade de destino")
    relacao: str = Field(min_length=1, description="Relacao em portugues")
    evidencia: str = Field(default="", description="Trecho curto de evidencia")


class StructuredRelationExtraction(BaseModel):
    relacoes: list[StructuredRelation] = Field(default_factory=list)


_PLACEHOLDER_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\b(?:pessoa|orgao|entidade|tema|assunto)\s+[a-z]\b", re.IGNORECASE),
    re.compile(r"\b(?:sujeito|objeto|entidade|tema)\s*\d+\b", re.IGNORECASE),
)

_THEME_HINTS = {"tema", "assunto", "topico", "subject", "subjecto"}

_PERSON_ROLE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bDep\.?\s+[A-Z][A-Za-zÀ-ÖØ-öø-ÿ'-]+\b"),
    re.compile(r"\bDeputad[oa]\s+[A-Z][A-Za-zÀ-ÖØ-öø-ÿ'-]+\b"),
)


def _preview(value: str, max_chars: int = MAX_DEBUG_CHARS) -> str:
    sanitized = value.replace("\n", "\\n")
    if len(sanitized) <= max_chars:
        return sanitized
    return sanitized[:max_chars] + "..."


def _set_last_extraction_debug(payload: dict[str, Any]) -> None:
    global _LAST_EXTRACTION_DEBUG
    _LAST_EXTRACTION_DEBUG = payload


def get_last_extraction_debug() -> dict[str, Any]:
    return _LAST_EXTRACTION_DEBUG


def _normalize_for_match(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    without_accents = "".join(char for char in normalized if not unicodedata.combining(char))
    lowered = without_accents.lower().replace("_", " ")
    lowered = re.sub(r"\s+", " ", lowered)
    return lowered.strip()


def _normalize_relation_pt_br(value: str) -> str:
    relation = "_".join(_normalize_for_match(value).split())
    relation = re.sub(r"\W+", "_", relation)
    relation = re.sub(r"_+", "_", relation).strip("_")
    return relation.lower() or "relaciona"


def _is_placeholder_value(value: str) -> bool:
    stripped = value.strip()
    if not stripped:
        return True

    normalized = _normalize_for_match(stripped)
    if normalized in {
        "pessoa x",
        "pessoa y",
        "orgao x",
        "orgao y",
        "entidade x",
        "entidade y",
        "tema",
        "tema discutido",
        "assunto",
        "assunto discutido",
    }:
        return True

    return any(pattern.search(stripped) for pattern in _PLACEHOLDER_PATTERNS)


def _source_contains_candidate(source_text: str, candidate: str) -> bool:
    normalized_source = _normalize_for_match(source_text)
    normalized_candidate = _normalize_for_match(candidate)
    if len(normalized_candidate) < 2:
        return False
    return normalized_candidate in normalized_source


def _label_from_category(category: str) -> str:
    normalized = _normalize_for_match(category)
    if any(hint in normalized for hint in _THEME_HINTS):
        return "TEMA"
    return "ENTIDADE"


def _build_entity_name_map(entities: list[Entity]) -> dict[str, str]:
    by_normalized: dict[str, str] = {}
    for entity in entities:
        normalized = _normalize_for_match(entity.name)
        if normalized and normalized not in by_normalized:
            by_normalized[normalized] = entity.name
    return by_normalized


def _resolve_entity_name(raw_name: str, entity_name_map: dict[str, str]) -> str | None:
    normalized = _normalize_for_match(raw_name)
    if not normalized:
        return None
    return entity_name_map.get(normalized)


def _extract_person_entities_from_text(text: str) -> list[Entity]:
    entities: list[Entity] = []
    seen: set[str] = set()
    for pattern in _PERSON_ROLE_PATTERNS:
        for match in pattern.finditer(text):
            name = match.group(0).strip()
            normalized = _normalize_for_match(name)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            entities.append(
                Entity(
                    name=name,
                    label="ENTIDADE",
                    properties={
                        "categoria": "PESSOA",
                        "contexto": "Pessoa identificada literalmente no texto por cargo e nome.",
                    },
                )
            )
    return entities


def configure_llamaindex() -> None:
    llm = Ollama(
        model=settings.ollama_model,
        base_url=settings.ollama_base_url,
        request_timeout=settings.ollama_timeout,
        temperature=0,
    )

    embed_model = OllamaEmbedding(
        model_name=settings.ollama_model,
        base_url=settings.ollama_base_url,
    )

    Settings.llm = llm
    Settings.embed_model = embed_model


class LlamaIndexKnowledgeExtractor(KnowledgeExtractor):
    def __init__(self) -> None:
        configure_llamaindex()

    def _extract_entities(self, text: str) -> tuple[list[Entity], str]:
        if Settings.llm is None:
            raise RuntimeError("Settings.llm is not configured")

        structured = Settings.llm.structured_predict(
            StructuredEntityExtraction,
            ENTITY_EXTRACTION_PROMPT,
            text=text,
        )

        entities_by_name: dict[str, Entity] = {}
        for item in structured.entidades:
            name = item.nome.strip()
            if not name:
                continue
            if _is_placeholder_value(name):
                continue
            if not _source_contains_candidate(text, name):
                continue

            existing = entities_by_name.get(name)
            candidate = Entity(
                name=name,
                label=_label_from_category(item.categoria),
                properties={
                    "categoria": item.categoria.strip(),
                    "contexto": item.contexto.strip(),
                },
            )

            # Prioriza ENTIDADE quando houver colisao entre labels.
            if existing is None or (existing.label == "TEMA" and candidate.label == "ENTIDADE"):
                entities_by_name[name] = candidate

        # Preserva mencoes literais de pessoas com cargo+nome (ex.: Dep. Marquito).
        for person_entity in _extract_person_entities_from_text(text):
            if person_entity.name not in entities_by_name:
                entities_by_name[person_entity.name] = person_entity

        preview = _preview(structured.model_dump_json(ensure_ascii=False))
        return list(entities_by_name.values()), preview

    def _extract_relationships(self, text: str, entities: list[Entity]) -> tuple[list[Relationship], str]:
        if Settings.llm is None:
            raise RuntimeError("Settings.llm is not configured")

        entities_payload = [
            {
                "nome": entity.name,
                "categoria": entity.properties.get("categoria", entity.label),
                "contexto": entity.properties.get("contexto", ""),
            }
            for entity in entities
        ]

        structured = Settings.llm.structured_predict(
            StructuredRelationExtraction,
            RELATION_EXTRACTION_PROMPT,
            text=text,
            entities_json=json.dumps(entities_payload, ensure_ascii=False),
        )

        entity_name_map = _build_entity_name_map(entities)
        relationships: list[Relationship] = []
        seen: set[tuple[str, str, str]] = set()

        for item in structured.relacoes:
            source = _resolve_entity_name(item.origem, entity_name_map)
            target = _resolve_entity_name(item.destino, entity_name_map)
            relation = _normalize_relation_pt_br(item.relacao)

            if not source or not target:
                continue
            if source == target:
                continue
            if not relation:
                continue

            key = (source, relation, target)
            if key in seen:
                continue
            seen.add(key)

            relationships.append(
                Relationship(
                    source=source,
                    target=target,
                    relation=relation,
                    properties={"evidencia": item.evidencia.strip()},
                )
            )

        preview = _preview(structured.model_dump_json(ensure_ascii=False))
        return relationships, preview

    def extract(self, text: str) -> KnowledgeGraphExtraction:
        attempts: list[dict[str, Any]] = []

        entities, entities_preview = self._extract_entities(text)
        attempts.append(
            {
                "attempt": "entities_structured_predict",
                "ok": bool(entities),
                "raw_preview": entities_preview,
                "entity_count": len(entities),
            }
        )

        if not entities:
            _set_last_extraction_debug(
                {
                    "pipeline": "llamaindex_structured_pydantic_two_step",
                    "attempts": attempts,
                    "fallback_used": True,
                }
            )
            return KnowledgeGraphExtraction(entities=[], relationships=[])

        relationships, relationships_preview = self._extract_relationships(text, entities)
        attempts.append(
            {
                "attempt": "relationships_structured_predict",
                "ok": True,
                "raw_preview": relationships_preview,
                "triplet_count": len(relationships),
            }
        )

        _set_last_extraction_debug(
            {
                "pipeline": "llamaindex_structured_pydantic_two_step",
                "attempts": attempts,
                "fallback_used": False,
            }
        )

        return KnowledgeGraphExtraction(entities=entities, relationships=relationships)
