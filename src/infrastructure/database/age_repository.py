from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import psycopg
from psycopg import sql

from config.settings import settings
from src.domain.models import Entity, KnowledgeGraphExtraction, Relationship
from src.domain.normalization import normalize_graph_category, normalize_graph_name, normalize_relation_label
from src.domain.repositories import GraphRepository
from src.infrastructure.database.connection import init_apache_age


def _safe_relation_label(value: str) -> str:
    normalized = re.sub(r"\W", "_", normalize_relation_label(value))
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


def _parse_agtype_json(value: object) -> Any:
    rendered = _clean_agtype_scalar(value)
    if not rendered:
        return {}
    try:
        return json.loads(rendered)
    except json.JSONDecodeError:
        return {}


@dataclass(frozen=True)
class _GraphNode:
    node_id: int
    name: str
    label: str
    properties: dict[str, Any]


@dataclass(frozen=True)
class _GraphEdge:
    source_id: int
    target_id: int
    relation: str
    properties: dict[str, Any]


@dataclass(frozen=True)
class _NormalizationSnapshot:
    canonical_nodes: list[_GraphNode]
    canonical_by_node_id: dict[int, _GraphNode]
    duplicate_ids: list[int]


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
        normalized_name = normalize_graph_name(entity.name)
        if not normalized_name:
            return

        normalized_label = normalize_graph_category(entity.label) or "ENTIDADE"
        normalized_properties = dict(entity.properties)
        normalized_properties["categoria"] = normalize_graph_category(
            str(normalized_properties.get("categoria", normalized_label))
        )

        cypher = sql.SQL(
            "SELECT * FROM cypher({}, $$ "
            "MERGE (n:Entidade {{name: $name}}) "
            "SET n.label = $label, n.properties = $properties "
            "RETURN n $$, %s::agtype) as (n agtype);"
        ).format(sql.Literal(settings.graph_name))
        params = {
            "name": normalized_name,
            "label": normalized_label,
            "properties": normalized_properties,
        }
        cursor.execute(cypher, (_to_agtype_params(params),))

    def _merge_relationship(self, cursor: psycopg.Cursor, relationship: Relationship) -> None:
        relation = _safe_relation_label(relationship.relation)
        normalized_source = normalize_graph_name(relationship.source)
        normalized_target = normalize_graph_name(relationship.target)
        if not normalized_source or not normalized_target:
            return

        cypher = sql.SQL(
            "SELECT * FROM cypher({}, $$ "
            "MERGE (a:Entidade {{name: $source}}) "
            "MERGE (b:Entidade {{name: $target}}) "
            "MERGE (a)-[r:{}]->(b) "
            "SET r.properties = $properties "
            "RETURN a, r, b $$, %s::agtype) as (a agtype, r agtype, b agtype);"
        ).format(sql.Literal(settings.graph_name), sql.SQL(relation))
        params = {
            "source": normalized_source,
            "target": normalized_target,
            "properties": relationship.properties,
        }
        cursor.execute(cypher, (_to_agtype_params(params),))

    def _fetch_graph_nodes(self) -> list[_GraphNode]:
        query = sql.SQL(
            "SELECT * FROM cypher({}, $$ "
            "MATCH (n:Entidade) "
            "RETURN id(n), coalesce(n.name, ''), coalesce(n.label, 'ENTIDADE'), coalesce(n.properties, {{}}) "
            "$$, %s::agtype) as (node_id agtype, name agtype, label agtype, properties agtype);"
        ).format(sql.Literal(settings.graph_name))
        with self._conn.cursor() as cursor:
            cursor.execute(query, (_to_agtype_params({}),))
            rows = cursor.fetchall()

        result: list[_GraphNode] = []
        for node_id, name, label, properties in rows:
            result.append(
                _GraphNode(
                    node_id=int(_clean_agtype_scalar(node_id)),
                    name=_clean_agtype_scalar(name),
                    label=_clean_agtype_scalar(label) or "ENTIDADE",
                    properties=_parse_agtype_json(properties),
                )
            )
        return result

    def _fetch_graph_edges(self) -> list[_GraphEdge]:
        query = sql.SQL(
            "SELECT * FROM cypher({}, $$ "
            "MATCH (a:Entidade)-[r]->(b:Entidade) "
            "RETURN id(a), type(r), id(b), coalesce(r.properties, {{}}) "
            "$$, %s::agtype) as (source_id agtype, relation agtype, target_id agtype, properties agtype);"
        ).format(sql.Literal(settings.graph_name))
        with self._conn.cursor() as cursor:
            cursor.execute(query, (_to_agtype_params({}),))
            rows = cursor.fetchall()

        result: list[_GraphEdge] = []
        for source_id, relation, target_id, properties in rows:
            result.append(
                _GraphEdge(
                    source_id=int(_clean_agtype_scalar(source_id)),
                    target_id=int(_clean_agtype_scalar(target_id)),
                    relation=_safe_relation_label(_clean_agtype_scalar(relation)),
                    properties=_parse_agtype_json(properties),
                )
            )
        return result

    def _delete_all_relationships(self, cursor: psycopg.Cursor) -> None:
        query = sql.SQL(
            "SELECT * FROM cypher({}, $$ "
            "MATCH (:Entidade)-[r]->(:Entidade) "
            "DELETE r "
            "RETURN 1 $$, %s::agtype) as (ignored agtype);"
        ).format(sql.Literal(settings.graph_name))
        cursor.execute(query, (_to_agtype_params({}),))

    def _delete_node_by_id(self, cursor: psycopg.Cursor, node_id: int) -> None:
        query = sql.SQL(
            "SELECT * FROM cypher({}, $$ "
            "MATCH (n:Entidade) "
            "WHERE id(n) = $node_id "
            "DELETE n "
            "RETURN 1 $$, %s::agtype) as (ignored agtype);"
        ).format(sql.Literal(settings.graph_name))
        cursor.execute(query, (_to_agtype_params({"node_id": node_id}),))

    def _update_node_by_id(
        self,
        cursor: psycopg.Cursor,
        *,
        node_id: int,
        name: str,
        label: str,
        properties: dict[str, Any],
    ) -> None:
        query = sql.SQL(
            "SELECT * FROM cypher({}, $$ "
            "MATCH (n:Entidade) "
            "WHERE id(n) = $node_id "
            "SET n.name = $name, n.label = $label, n.properties = $properties "
            "RETURN n $$, %s::agtype) as (n agtype);"
        ).format(sql.Literal(settings.graph_name))
        params = {
            "node_id": node_id,
            "name": name,
            "label": label,
            "properties": properties,
        }
        cursor.execute(query, (_to_agtype_params(params),))

    def _merge_edge_by_ids(
        self,
        cursor: psycopg.Cursor,
        *,
        source_id: int,
        target_id: int,
        relation: str,
        properties: dict[str, Any],
    ) -> None:
        query = sql.SQL(
            "SELECT * FROM cypher({}, $$ "
            "MATCH (a:Entidade), (b:Entidade) "
            "WHERE id(a) = $source_id AND id(b) = $target_id "
            "MERGE (a)-[r:{}]->(b) "
            "SET r.properties = $properties "
            "RETURN r $$, %s::agtype) as (r agtype);"
        ).format(sql.Literal(settings.graph_name), sql.SQL(_safe_relation_label(relation)))
        params = {
            "source_id": source_id,
            "target_id": target_id,
            "properties": properties,
        }
        cursor.execute(query, (_to_agtype_params(params),))

    def _build_normalization_snapshot(self, nodes: list[_GraphNode]) -> _NormalizationSnapshot:
        groups: dict[str, list[_GraphNode]] = defaultdict(list)
        for node in nodes:
            normalized_name = normalize_graph_name(node.name)
            if not normalized_name:
                continue
            groups[normalized_name].append(node)

        canonical_by_node_id: dict[int, _GraphNode] = {}
        canonical_nodes: list[_GraphNode] = []
        duplicate_ids: list[int] = []

        for normalized_name, group in groups.items():
            keeper = min(group, key=lambda item: item.node_id)
            canonical_node = self._build_canonical_node(keeper, normalized_name)
            canonical_nodes.append(canonical_node)

            for node in group:
                canonical_by_node_id[node.node_id] = canonical_node
                if node.node_id != keeper.node_id:
                    duplicate_ids.append(node.node_id)

        return _NormalizationSnapshot(
            canonical_nodes=canonical_nodes,
            canonical_by_node_id=canonical_by_node_id,
            duplicate_ids=duplicate_ids,
        )

    def _build_canonical_node(self, keeper: _GraphNode, normalized_name: str) -> _GraphNode:
        merged_properties = dict(keeper.properties)
        merged_properties["categoria"] = normalize_graph_category(
            str(keeper.properties.get("categoria", keeper.label or "ENTIDADE"))
        )
        return _GraphNode(
            node_id=keeper.node_id,
            name=normalized_name,
            label=normalize_graph_category(keeper.label or "ENTIDADE") or "ENTIDADE",
            properties=merged_properties,
        )

    def _rebuild_canonical_edges(
        self,
        edges: list[_GraphEdge],
        canonical_by_node_id: dict[int, _GraphNode],
    ) -> dict[tuple[int, str, int], dict[str, Any]]:
        reconstructed_edges: dict[tuple[int, str, int], dict[str, Any]] = {}
        for edge in edges:
            source_node = canonical_by_node_id.get(edge.source_id)
            target_node = canonical_by_node_id.get(edge.target_id)
            if source_node is None or target_node is None:
                continue

            relation = _safe_relation_label(edge.relation)
            edge_key = (source_node.node_id, relation, target_node.node_id)
            if edge_key not in reconstructed_edges:
                reconstructed_edges[edge_key] = edge.properties
        return reconstructed_edges

    def _apply_graph_normalization(
        self,
        *,
        canonical_nodes: list[_GraphNode],
        duplicate_ids: list[int],
        reconstructed_edges: dict[tuple[int, str, int], dict[str, Any]],
    ) -> None:
        with self._conn.cursor() as cursor:
            self._delete_all_relationships(cursor)

            for node in canonical_nodes:
                self._update_node_by_id(
                    cursor,
                    node_id=node.node_id,
                    name=node.name,
                    label=node.label,
                    properties=node.properties,
                )

            for duplicate_id in duplicate_ids:
                self._delete_node_by_id(cursor, duplicate_id)

            for (source_id, relation, target_id), properties in reconstructed_edges.items():
                self._merge_edge_by_ids(
                    cursor,
                    source_id=source_id,
                    target_id=target_id,
                    relation=relation,
                    properties=properties,
                )

    def normalize_and_unify_graph_entities(self) -> dict[str, int]:
        nodes = self._fetch_graph_nodes()
        edges = self._fetch_graph_edges()

        if not nodes:
            return {
                "nodes_before": 0,
                "nodes_after": 0,
                "duplicates_removed": 0,
                "relationships_before": len(edges),
                "relationships_after": 0,
            }

        snapshot = self._build_normalization_snapshot(nodes)
        reconstructed_edges = self._rebuild_canonical_edges(edges, snapshot.canonical_by_node_id)
        self._apply_graph_normalization(
            canonical_nodes=snapshot.canonical_nodes,
            duplicate_ids=snapshot.duplicate_ids,
            reconstructed_edges=reconstructed_edges,
        )

        return {
            "nodes_before": len(nodes),
            "nodes_after": len(snapshot.canonical_nodes),
            "duplicates_removed": len(snapshot.duplicate_ids),
            "relationships_before": len(edges),
            "relationships_after": len(reconstructed_edges),
        }

    def search_graph(self, keyword: str, limit: int = 20) -> list[dict[str, str]]:
        query = sql.SQL(
            "SELECT * FROM cypher({}, $$ "
            "MATCH (a:Entidade)-[r]->(b:Entidade) "
            "WHERE toLower(a.name) CONTAINS toLower($kw) "
            "   OR toLower(b.name) CONTAINS toLower($kw) "
            "   OR toLower(coalesce(a.properties.contexto, '')) CONTAINS toLower($kw) "
            "   OR toLower(coalesce(a.properties.categoria, '')) CONTAINS toLower($kw) "
            "   OR toLower(coalesce(b.properties.contexto, '')) CONTAINS toLower($kw) "
            "   OR toLower(coalesce(b.properties.categoria, '')) CONTAINS toLower($kw) "
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
