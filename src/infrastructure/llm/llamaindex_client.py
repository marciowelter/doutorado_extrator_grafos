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
LLM_NOT_CONFIGURED_ERROR = "Settings.llm is not configured"
_LAST_EXTRACTION_DEBUG: dict[str, Any] = {
    "attempts": [],
    "fallback_used": False,
}


ENTITY_EXTRACTION_PROMPT = PromptTemplate(
    """
Voce e um especialista em extracao de entidades partir de textos fornecidos em portugues brasileiro.
Extraia entidades do texto e devolva SOMENTE JSON valido seguindo o schema.

Regras obrigatorias:
1) Nao invente entidades.
2) Use o nome da entidade exatamente como aparece no texto, quando possivel.
2.1) Nomes próprios devem ser considerados entidades.
2.2) Especial atenção para quando houver Dep. ou Deputado, pois geralmente indicam nomes de pessoas importantes.
3) Para categoria, use valores como: PESSOA, ORGANIZACAO, LOCAL, DATA, VALOR, TEMA, EVENTO.
4) contexto deve ser curto (uma frase) explicando por que a entidade e relevante.
5) Se nao houver entidades, retorne lista vazia.

Texto:
{text}
""".strip()
)


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


RELATION_EXTRACTION_PROMPT = PromptTemplate(
    """
Voce e um especialista em extracao de nos e relacoes em portugues brasileiro.
Com base no texto e na lista de nos previamente extraidos (temas e entidades), identifique as relacoes que conectam esses nos com base no contexto do texto analisado.
Devolva SOMENTE JSON valido seguindo o schema.

Regras obrigatorias:
1) Use somente nos existentes na lista fornecida.
2) origem e destino devem ser os nomes exatos dos nos fornecidos.
3) relacao e opcional. Quando existir, use EXATAMENTE UM UNICO VERBO em portugues, sem preposicoes, sem complementos, em MAIUSCULAS (ex.: CUMPRIMENTOU, APROVOU, REPRESENTA, PERTENCE, PRODUZIU, INTEGRA, CITA).
4) evidencia e opcional. Quando existir, use um trecho curto do texto que sustenta a relacao.
5) E permitido retornar relacoes sem tipo e sem evidencia (relacao: null e evidencia: null), mantendo apenas a conexao entre nos.
6) Todos os nos devem possuir relacao entre si, ou seja, nao devem existir nos isolados sem conexao com outros.

Texto:
{text}

Nos previamente extraidos (JSON):
{nodes_json}
""".strip()
)


class StructuredEntity(BaseModel):
    nome: str = Field(min_length=1, description="Nome exato da entidade")
    categoria: str = Field(min_length=1, description="Categoria da entidade")
    contexto: str | None = Field(default=None, description="Resumo curto da relevancia (opcional)")


class StructuredEntityExtraction(BaseModel):
    entidades: list[StructuredEntity] = Field(default_factory=list)


class StructuredTheme(BaseModel):
    nome: str = Field(min_length=1, description="Nome curto do tema")
    contexto: str | None = Field(default=None, description="Resumo curto da relevancia (opcional)")


class StructuredThemeExtraction(BaseModel):
    temas: list[StructuredTheme] = Field(default_factory=list)


class StructuredRelation(BaseModel):
    origem: str = Field(min_length=1, description="Entidade de origem")
    destino: str = Field(min_length=1, description="Entidade de destino")
    relacao: str | None = Field(
        default=None,
        description="Um unico verbo em MAIUSCULAS (ex.: APROVOU, PRODUZIU) (opcional)",
    )
    evidencia: str | None = Field(default=None, description="Trecho curto de evidencia (opcional)")


class StructuredRelationExtraction(BaseModel):
    relacoes: list[StructuredRelation] = Field(default_factory=list)


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
    normalized = unicodedata.normalize("NFKD", value)
    without_accents = "".join(char for char in normalized if not unicodedata.combining(char))
    lowered = without_accents.lower().replace("_", " ")
    lowered = re.sub(r"\s+", " ", lowered)
    return lowered.strip()


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


def _resolve_entity_name(raw_name: str, entity_name_map: dict[str, str]) -> str | None:
    normalized = _normalize_for_match(raw_name)
    if not normalized:
        return None
    return entity_name_map.get(normalized)


def _sanitize_relation_label(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-zÀ-ÿ_ ]", " ", value).strip()
    if not cleaned:
        return "RELACIONA"

    token = cleaned.split()[0]
    normalized = _normalize_for_match(token)
    if not normalized:
        return "RELACIONA"

    return normalized.replace(" ", "_").upper()


def _parse_structured_relationship_item(
    item: StructuredRelation,
    entity_name_map: dict[str, str],
) -> tuple[Relationship, tuple[str, str], tuple[str, str, str]] | None:
    source = _resolve_entity_name(item.origem, entity_name_map)
    target = _resolve_entity_name(item.destino, entity_name_map)
    relation = _sanitize_relation_label((item.relacao or "").strip())

    if not source or not target:
        return None

    properties: dict[str, str] = {}
    evidence = (item.evidencia or "").strip()
    if evidence:
        properties["evidencia"] = evidence

    relationship = Relationship(
        source=source,
        target=target,
        relation=relation,
        properties=properties,
    )
    pair = (source, target)
    dedupe_key = (source, _normalize_for_match(relation), target)
    return relationship, pair, dedupe_key


def _ensure_complete_relationship_mesh(
    entities: list[Entity],
    connected_pairs: set[tuple[str, str]],
    relationships: list[Relationship],
) -> None:
    # Garante cobertura todos-para-todos entre entidades extraidas.
    for source in entities:
        for target in entities:
            if source.name == target.name:
                continue
            pair = (source.name, target.name)
            if pair in connected_pairs:
                continue
            connected_pairs.add(pair)
            relationships.append(
                Relationship(
                    source=source.name,
                    target=target.name,
                    relation="RELACIONA",
                    properties={},
                )
            )


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

    def _extract_themes(self, text: str) -> tuple[list[Entity], str]:
        if Settings.llm is None:
            raise RuntimeError(LLM_NOT_CONFIGURED_ERROR)

        structured = Settings.llm.structured_predict(
            StructuredThemeExtraction,
            THEME_EXTRACTION_PROMPT,
            text=text,
        )

        themes_by_name: dict[str, Entity] = {}
        for item in structured.temas:
            name = item.nome.strip()
            if not name:
                continue
            if not _source_contains_candidate(text, name):
                continue

            themes_by_name[name] = Entity(
                name=name,
                label="TEMA",
                properties={
                    "categoria": "TEMA",
                    "contexto": (item.contexto or "").strip(),
                },
            )

        preview = _preview(structured.model_dump_json(ensure_ascii=False))
        return list(themes_by_name.values()), preview

    def _extract_entities(self, text: str) -> tuple[list[Entity], str]:
        if Settings.llm is None:
            raise RuntimeError(LLM_NOT_CONFIGURED_ERROR)

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
            if not _source_contains_candidate(text, name):
                continue

            category = item.categoria.strip()
            if _label_from_category(category) == "TEMA":
                # Temas sao tratados em etapa dedicada para manter separacao semantica.
                continue

            candidate = Entity(
                name=name,
                label="ENTIDADE",
                properties={
                    "categoria": category,
                    "contexto": (item.contexto or "").strip(),
                },
            )

            entities_by_name[name] = candidate

        preview = _preview(structured.model_dump_json(ensure_ascii=False))
        return list(entities_by_name.values()), preview

    def _extract_relationships(self, text: str, nodes: list[Entity]) -> tuple[list[Relationship], str]:
        if Settings.llm is None:
            raise RuntimeError(LLM_NOT_CONFIGURED_ERROR)

        nodes_payload = [
            {
                "nome": node.name,
                "tipo": node.label,
                "categoria": node.properties.get("categoria", node.label),
                "contexto": node.properties.get("contexto", ""),
            }
            for node in nodes
        ]

        structured = Settings.llm.structured_predict(
            StructuredRelationExtraction,
            RELATION_EXTRACTION_PROMPT,
            text=text,
            nodes_json=json.dumps(nodes_payload, ensure_ascii=False),
        )

        entity_name_map = _build_entity_name_map(nodes)
        relationships: list[Relationship] = []
        seen: set[tuple[str, str, str]] = set()
        connected_pairs: set[tuple[str, str]] = set()

        for item in structured.relacoes:
            parsed = _parse_structured_relationship_item(item, entity_name_map)
            if parsed is None:
                continue
            relationship, pair, key = parsed
            if key in seen:
                continue
            seen.add(key)
            connected_pairs.add(pair)
            relationships.append(relationship)

        _ensure_complete_relationship_mesh(nodes, connected_pairs, relationships)

        preview = _preview(structured.model_dump_json(ensure_ascii=False))
        return relationships, preview

    def extract(self, text: str) -> KnowledgeGraphExtraction:
        attempts: list[dict[str, Any]] = []

        themes, themes_preview = self._extract_themes(text)
        attempts.append(
            {
                "attempt": "themes_structured_predict",
                "ok": bool(themes),
                "raw_preview": themes_preview,
                "theme_count": len(themes),
            }
        )

        entities, entities_preview = self._extract_entities(text)
        attempts.append(
            {
                "attempt": "entities_structured_predict",
                "ok": bool(entities),
                "raw_preview": entities_preview,
                "entity_count": len(entities),
            }
        )

        nodes = _merge_nodes(themes, entities)

        if not nodes:
            _set_last_extraction_debug(
                {
                    "pipeline": "llamaindex_structured_pydantic_three_step",
                    "attempts": attempts,
                    "fallback_used": True,
                }
            )
            return KnowledgeGraphExtraction(entities=[], relationships=[])

        relationships, relationships_preview = self._extract_relationships(text, nodes)
        attempts.append(
            {
                "attempt": "relationships_structured_predict",
                "ok": True,
                "raw_preview": relationships_preview,
                "triplet_count": len(relationships),
            }
        )

        # Mantem categorias e relacoes como retornadas pelo LLM,
        # aplicando apenas filtros basicos durante a extracao.

        _set_last_extraction_debug(
            {
                "pipeline": "llamaindex_structured_pydantic_three_step",
                "attempts": attempts,
                "fallback_used": False,
            }
        )

        return KnowledgeGraphExtraction(entities=nodes, relationships=relationships)
