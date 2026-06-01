from __future__ import annotations

import time
from typing import Any

from gliner import GLiNER
from llama_index.core import Settings
from llama_index.core.prompts import PromptTemplate
from llama_index.llms.ollama import Ollama
from pydantic import BaseModel, Field

from config.settings import settings
from src.domain.models import Entity, KnowledgeGraphExtraction, Relationship
from src.domain.normalization import (
    normalize_for_match,
    normalize_graph_category,
    normalize_graph_name,
)
from src.domain.repositories import KnowledgeExtractor


MAX_DEBUG_CHARS = 400
LLM_NOT_CONFIGURED_ERROR = "Settings.llm is not configured"
_LAST_EXTRACTION_DEBUG: dict[str, Any] = {"attempts": [], "fallback_used": False}


THEME_EXTRACTION_PROMPT = PromptTemplate(
    """
Voce e um especialista em extracao de temas e assuntos em textos de portugues brasileiro.
Extraia uma lista de temas tratados no texto e devolva SOMENTE JSON valido seguindo o schema.

Regras obrigatorias:
1) Nao invente temas; use apenas o que estiver no texto.
2) O nome do tema deve ser curto e claro (uma frase curta ou sintagma nominal), por exemplo: EDUCAÇÃO, SAÚDE, AGRICULTURA, SEGURANÇA PÚBLICA, MEIO AMBIENTE, CULTURA.
3) contexto deve ser curto (uma frase) explicando por que o tema e relevante no texto.
4) Se nao houver temas identificaveis, retorne lista vazia.

Texto:
{text}
""".strip()
)


class StructuredTheme(BaseModel):
    nome: str = Field(min_length=1, description="Nome curto do tema")
    contexto: str | None = Field(default=None, description="Resumo curto da relevancia (opcional)")


class StructuredThemeExtraction(BaseModel):
    temas: list[StructuredTheme] = Field(default_factory=list)


_THEME_HINTS = {"tema", "assunto", "topico", "subject", "subjecto"}


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
    return normalize_for_match(value)


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


def _merge_nodes(themes: list[Entity], entities: list[Entity]) -> list[Entity]:
    merged_by_name: dict[str, Entity] = {}

    # Prioriza temas quando o mesmo nome aparecer em ambos.
    for node in [*themes, *entities]:
        normalized = _normalize_for_match(node.name)
        if not normalized:
            continue
        if normalized in merged_by_name:
            continue
        merged_by_name[normalized] = node

    return list(merged_by_name.values())


def _parse_gliner_labels(raw_value: str) -> list[str]:
    labels = [part.strip() for part in raw_value.split(",") if part.strip()]
    return labels or ["pessoa", "organizacao", "local", "data", "evento", "valor"]


def _extract_entity_context(text: str, start: int | None, end: int | None, radius: int = 60) -> str:
    if start is None or end is None:
        return ""
    left = max(0, start - radius)
    right = min(len(text), end + radius)
    snippet = text[left:right].strip()
    snippet = " ".join(snippet.split())
    return snippet


def _build_fixed_relationships(entities: list[Entity], themes: list[Entity]) -> list[Relationship]:
    relationships: list[Relationship] = []
    seen: set[tuple[str, str]] = set()

    for entity in entities:
        for theme in themes:
            if entity.name == theme.name:
                continue
            key = (entity.name, theme.name)
            if key in seen:
                continue
            seen.add(key)
            relationships.append(
                Relationship(
                    source=entity.name,
                    target=theme.name,
                    relation="RELACIONA",
                    properties={},
                )
            )

    return relationships


def _build_theme_entity(raw_name: str, contexto: str) -> Entity | None:
    name = normalize_graph_name(raw_name.strip())
    if not name:
        return None

    return Entity(
        name=name,
        label="TEMA",
        properties={
            "categoria": "TEMA",
            "contexto": contexto,
        },
    )


def configure_llamaindex() -> None:
    llm = Ollama(
        model=settings.ollama_model,
        base_url=settings.ollama_base_url,
        request_timeout=settings.ollama_timeout,
        temperature=0,
    )

    Settings.llm = llm


class LlamaIndexKnowledgeExtractor(KnowledgeExtractor):
    def __init__(self) -> None:
        configure_llamaindex()
        self._ner_model = GLiNER.from_pretrained(settings.gliner_model)
        self._gliner_labels = _parse_gliner_labels(settings.gliner_labels)

    def _extract_themes(
        self,
        text: str,
        additional_themes: list[str] | None = None,
    ) -> tuple[list[Entity], str]:
        if Settings.llm is None:
            raise RuntimeError(LLM_NOT_CONFIGURED_ERROR)

        structured = Settings.llm.structured_predict(
            StructuredThemeExtraction,
            THEME_EXTRACTION_PROMPT,
            text=text,
        )

        themes_by_name: dict[str, Entity] = {}
        for item in structured.temas:
            theme_entity = _build_theme_entity(item.nome, (item.contexto or "").strip())
            if theme_entity is None:
                continue
            themes_by_name[theme_entity.name] = theme_entity

        if additional_themes:
            for raw_theme in additional_themes:
                theme_entity = _build_theme_entity(
                    str(raw_theme),
                    "Tema adicionado a partir da tabela datamart_oque.",
                )
                if theme_entity is None:
                    continue
                if theme_entity.name in themes_by_name:
                    continue
                themes_by_name[theme_entity.name] = theme_entity

        preview = _preview(structured.model_dump_json(ensure_ascii=False))
        return list(themes_by_name.values()), preview

    def _extract_entities(self, text: str) -> tuple[list[Entity], str]:
        gliner_result = self._ner_model.predict_entities(
            text,
            labels=self._gliner_labels,
            threshold=settings.gliner_threshold,
        )

        entities_by_name: dict[str, Entity] = {}
        debug_items: list[dict[str, Any]] = []

        for raw in gliner_result:
            raw_name = str(raw.get("text", "")).strip()
            if not raw_name:
                continue
            if not _source_contains_candidate(text, raw_name):
                continue

            raw_label = str(raw.get("label", "ENTIDADE")).strip()
            category = normalize_graph_category(raw_label)
            if _label_from_category(category) == "TEMA":
                # Temas continuam em etapa dedicada via Ollama.
                continue

            name = normalize_graph_name(raw_name)
            if not name:
                continue

            score = raw.get("score")
            start = raw.get("start")
            end = raw.get("end")
            contexto = _extract_entity_context(
                text,
                int(start) if isinstance(start, int | float) else None,
                int(end) if isinstance(end, int | float) else None,
            )

            properties: dict[str, Any] = {
                "categoria": category,
                "contexto": contexto,
            }
            if isinstance(score, int | float):
                properties["score"] = round(float(score), 4)

            entities_by_name[name] = Entity(
                name=name,
                label="ENTIDADE",
                properties=properties,
            )

            debug_items.append(
                {
                    "text": raw_name,
                    "label": raw_label,
                    "score": score,
                    "start": start,
                    "end": end,
                }
            )

        preview = _preview(str(debug_items))
        return list(entities_by_name.values()), preview

    def extract(
        self,
        text: str,
        additional_themes: list[str] | None = None,
    ) -> KnowledgeGraphExtraction:
        started_at = time.perf_counter()
        attempts: list[dict[str, Any]] = []

        themes_started_at = time.perf_counter()
        themes, themes_preview = self._extract_themes(text, additional_themes=additional_themes)
        themes_seconds = round(time.perf_counter() - themes_started_at, 4)
        attempts.append(
            {
                "attempt": "themes_structured_predict",
                "ok": bool(themes),
                "raw_preview": themes_preview,
                "theme_count": len(themes),
                "additional_theme_count": len(additional_themes or []),
                "duration_seconds": themes_seconds,
            }
        )

        entities_started_at = time.perf_counter()
        entities, entities_preview = self._extract_entities(text)
        entities_seconds = round(time.perf_counter() - entities_started_at, 4)
        attempts.append(
            {
                "attempt": "entities_gliner_predict",
                "ok": bool(entities),
                "raw_preview": entities_preview,
                "entity_count": len(entities),
                "duration_seconds": entities_seconds,
            }
        )

        nodes = _merge_nodes(themes, entities)

        if not nodes:
            _set_last_extraction_debug(
                {
                    "pipeline": "gliner_entities_ollama_themes_fixed_relationships",
                    "attempts": attempts,
                    "timings": {
                        "themes_ollama_seconds": themes_seconds,
                        "entities_gliner_seconds": entities_seconds,
                        "relationships_build_seconds": 0.0,
                        "extraction_total_seconds": round(time.perf_counter() - started_at, 4),
                    },
                    "fallback_used": True,
                }
            )
            return KnowledgeGraphExtraction(entities=[], relationships=[])

        relationships_started_at = time.perf_counter()
        relationships = _build_fixed_relationships(entities=entities, themes=themes)
        relationships_seconds = round(time.perf_counter() - relationships_started_at, 4)
        attempts.append(
            {
                "attempt": "relationships_cartesian_entities_themes",
                "ok": True,
                "raw_preview": f"entities={len(entities)} themes={len(themes)}",
                "triplet_count": len(relationships),
                "duration_seconds": relationships_seconds,
            }
        )

        _set_last_extraction_debug(
            {
                "pipeline": "gliner_entities_ollama_themes_fixed_relationships",
                "attempts": attempts,
                "timings": {
                    "themes_ollama_seconds": themes_seconds,
                    "entities_gliner_seconds": entities_seconds,
                    "relationships_build_seconds": relationships_seconds,
                    "extraction_total_seconds": round(time.perf_counter() - started_at, 4),
                },
                "fallback_used": False,
            }
        )

        return KnowledgeGraphExtraction(entities=nodes, relationships=relationships)
