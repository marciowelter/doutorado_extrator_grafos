from __future__ import annotations

import json
import re

import psycopg
from psycopg import sql

from config.settings import settings
from src.domain.models import Entity, KnowledgeGraphExtraction, Relationship
from src.domain.repositories import GraphRepository
from src.infrastructure.database.connection import init_apache_age


def _safe_relation_label(value: str) -> str:
    normalized = re.sub(r"\W", "_", value.upper())
    return normalized or "RELACIONA"


def _to_agtype_params(params: dict) -> str:
    return json.dumps(params, ensure_ascii=False)


def _clean_agtype_scalar(value: object) -> str:
    rendered = str(value).strip()
    if rendered.endswith("::agtype"):
        rendered = rendered[: -len("::agtype")].strip()
    if len(rendered) >= 2 and rendered[0] == '"' and rendered[-1] == '"':
        rendered = rendered[1:-1]
    return rendered


class AgeRepository(GraphRepository):
    def __init__(self, conn: psycopg.Connection):
        self._conn = conn

    def ensure_graph(self) -> None:
        init_apache_age(self._conn)

    def save_extraction(self, extraction: KnowledgeGraphExtraction) -> None:
        with self._conn.cursor() as cursor:
            for entity in extraction.entities:
                self._merge_entity(cursor, entity)
            for relationship in extraction.relationships:
                self._merge_relationship(cursor, relationship)

    def _merge_entity(self, cursor: psycopg.Cursor, entity: Entity) -> None:
        cypher = sql.SQL(
            "SELECT * FROM cypher({}, $$ "
            "MERGE (n:Entidade {{name: $name}}) "
            "SET n.label = $label, n.properties = $properties "
            "RETURN n $$, %s::agtype) as (n agtype);"
        ).format(sql.Literal(settings.graph_name))
        params = {
            "name": entity.name,
            "label": entity.label,
            "properties": entity.properties,
        }
        cursor.execute(cypher, (_to_agtype_params(params),))

    def _merge_relationship(self, cursor: psycopg.Cursor, relationship: Relationship) -> None:
        relation = _safe_relation_label(relationship.relation)
        cypher = sql.SQL(
            "SELECT * FROM cypher({}, $$ "
            "MERGE (a:Entidade {{name: $source}}) "
            "MERGE (b:Entidade {{name: $target}}) "
            "MERGE (a)-[r:{}]->(b) "
            "SET r.properties = $properties "
            "RETURN a, r, b $$, %s::agtype) as (a agtype, r agtype, b agtype);"
        ).format(sql.Literal(settings.graph_name), sql.SQL(relation))
        params = {
            "source": relationship.source,
            "target": relationship.target,
            "properties": relationship.properties,
        }
        cursor.execute(cypher, (_to_agtype_params(params),))

    def search_graph(self, keyword: str, limit: int = 20) -> list[dict[str, str]]:
        query = sql.SQL(
            "SELECT * FROM cypher({}, $$ "
            "MATCH (a:Entidade)-[r]->(b:Entidade) "
            "WHERE toLower(a.name) CONTAINS toLower($kw) "
            "   OR toLower(b.name) CONTAINS toLower($kw) "
            "RETURN a.name, coalesce(a.label, 'ENTIDADE'), type(r), b.name, coalesce(b.label, 'ENTIDADE') "
            "LIMIT $limit $$, %s::agtype) "
            "as (source agtype, source_label agtype, relation agtype, target agtype, target_label agtype);"
        ).format(sql.Literal(settings.graph_name))
        params = {"kw": keyword, "limit": limit}
        with self._conn.cursor() as cursor:
            cursor.execute(query, (_to_agtype_params(params),))
            rows = cursor.fetchall()

        result: list[dict[str, str]] = []
        for source, source_label, relation, target, target_label in rows:
            result.append(
                {
                    "source": _clean_agtype_scalar(source),
                    "source_label": _clean_agtype_scalar(source_label),
                    "relation": _clean_agtype_scalar(relation),
                    "target": _clean_agtype_scalar(target),
                    "target_label": _clean_agtype_scalar(target_label),
                }
            )
        return result
