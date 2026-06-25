#!/usr/bin/env python3
"""Executa fases de migração grafo legado -> hub por discurso."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import settings
from src.infrastructure.database.connection import get_postgres_connection, init_apache_age


def _clean_agtype(value: object) -> str:
    rendered = str(value).strip()
    if rendered.endswith("::agtype"):
        rendered = rendered[: -len("::agtype")].strip()
    if len(rendered) >= 2 and rendered[0] == '"' and rendered[-1] == '"':
        rendered = rendered[1:-1]
    return rendered


def _cypher_count(conn, cypher_body: str) -> int:
    graph = settings.graph_name
    query = (
        f"SELECT * FROM cypher('{graph}', $$ "
        f"{cypher_body} "
        f"$$) AS (total agtype);"
    )
    with conn.cursor() as cursor:
        cursor.execute("LOAD 'age';")
        cursor.execute("SET search_path = ag_catalog, '$user', public;")
        cursor.execute(query)
        row = cursor.fetchone()
    return int(_clean_agtype(row[0])) if row else 0


def _cypher_delete_relaciona_batch(conn, batch_size: int = 50000) -> int:
    graph = settings.graph_name
    total_deleted = 0
    while True:
        query = (
            f"SELECT * FROM cypher('{graph}', $$ "
            "MATCH (:Entidade)-[r:RELACIONA]->(:Entidade) "
            "WITH r LIMIT $limit "
            "DELETE r "
            "RETURN count(r) "
            "$$) AS (deleted agtype);"
        )
        with conn.cursor() as cursor:
            cursor.execute("LOAD 'age';")
            cursor.execute("SET search_path = ag_catalog, '$user', public;")
            cursor.execute(
                query.replace("$limit", str(batch_size)),
            )
            row = cursor.fetchone()
        deleted = int(_clean_agtype(row[0])) if row else 0
        total_deleted += deleted
        print(f"deleted_batch={deleted} total_deleted={total_deleted}", flush=True)
        if deleted == 0:
            break
    return total_deleted


def _cypher_delete_relaciona(conn) -> int:
    return _cypher_delete_relaciona_batch(conn)


def audit_baseline(datamart_conn, graph_conn) -> dict[str, int]:
    with datamart_conn.cursor() as cursor:
        cursor.execute(
            "SELECT COUNT(*) FROM doutorado.datamart_trecho WHERE length(texto) > 1000"
        )
        total_elegiveis = int(cursor.fetchone()[0])

        cursor.execute(
            "SELECT COUNT(*) FROM doutorado.datamart_trecho "
            "WHERE grafo IS TRUE AND length(texto) > 1000"
        )
        ja_processados = int(cursor.fetchone()[0])

        cursor.execute(
            "SELECT COUNT(*) FROM doutorado.datamart_trecho "
            "WHERE (grafo IS FALSE OR grafo IS NULL) AND length(texto) > 1000"
        )
        pendentes = int(cursor.fetchone()[0])

    init_apache_age(graph_conn)
    with graph_conn.cursor() as cursor:
        cursor.execute("LOAD 'age';")
        cursor.execute("SET search_path = ag_catalog, '$user', public;")
        cursor.execute(
            f'SELECT count(*) FROM {settings.graph_name}."Entidade"'
        )
        total_nodes = int(cursor.fetchone()[0])

    return {
        "total_elegiveis": total_elegiveis,
        "ja_processados": ja_processados,
        "pendentes": pendentes,
        "total_nodes": total_nodes,
        "relaciona_edges": _cypher_count(
            graph_conn,
            "MATCH (:Entidade)-[r:RELACIONA]->(:Entidade) RETURN count(r)",
        ),
        "ocorre_em_edges": _cypher_count(
            graph_conn,
            "MATCH (:Entidade)-[r:OCORRE_EM]->(:Entidade) RETURN count(r)",
        ),
        "discurso_nodes": _cypher_count(
            graph_conn,
            "MATCH (n:Entidade) WHERE n.label = 'DISCURSO' RETURN count(n)",
        ),
    }


def delete_relaciona(graph_conn) -> int:
    init_apache_age(graph_conn)
    return _cypher_delete_relaciona(graph_conn)


def reset_grafo_flags(datamart_conn) -> int:
    with datamart_conn.cursor() as cursor:
        cursor.execute(
            "UPDATE doutorado.datamart_trecho "
            "SET grafo = FALSE "
            "WHERE grafo IS TRUE AND length(texto) > 1000"
        )
        updated = cursor.rowcount

        cursor.execute(
            "SELECT COUNT(*) FROM doutorado.datamart_trecho "
            "WHERE (grafo IS FALSE OR grafo IS NULL) AND length(texto) > 1000"
        )
        pendentes = int(cursor.fetchone()[0])
    return {"updated": updated, "pendentes": pendentes}


def validate(datamart_conn, graph_conn) -> dict[str, int | list]:
    baseline = audit_baseline(datamart_conn, graph_conn)

    graph_conn2 = get_postgres_connection()
    init_apache_age(graph_conn2)
    sample: list[dict[str, str]] = []
    query = (
        f"SELECT * FROM cypher('{settings.graph_name}', $$ "
        "MATCH (n:Entidade)-[:OCORRE_EM]->(d:Entidade) "
        "WHERE d.label = 'DISCURSO' "
        "RETURN n.name, n.label, d.name, d.properties.data_ocorrencia "
        "LIMIT 5 "
        "$$) AS (source agtype, source_label agtype, discurso agtype, data agtype);"
    )
    with graph_conn2.cursor() as cursor:
        cursor.execute("LOAD 'age';")
        cursor.execute("SET search_path = ag_catalog, '$user', public;")
        cursor.execute(query)
        for source, source_label, discurso, data in cursor.fetchall():
            sample.append(
                {
                    "source": _clean_agtype(source),
                    "source_label": _clean_agtype(source_label),
                    "discurso": _clean_agtype(discurso),
                    "data_ocorrencia": _clean_agtype(data),
                }
            )
    graph_conn2.close()

    return {
        **baseline,
        "sample_ocorre_em": sample,
        "ok": baseline["relaciona_edges"] == 0 and baseline["ocorre_em_edges"] > 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Migracao grafo legado -> hub discurso")
    parser.add_argument(
        "phase",
        choices=("audit", "delete-relaciona", "reset-grafo", "validate", "prepare"),
        help="audit=baseline; prepare=delete+reset; validate=pos-migracao",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Arquivo JSON opcional para salvar resultado",
    )
    args = parser.parse_args()

    datamart_conn = get_postgres_connection(dbname="banco", schema="doutorado")
    graph_conn = get_postgres_connection()

    try:
        if args.phase == "audit":
            result = audit_baseline(datamart_conn, graph_conn)
        elif args.phase == "delete-relaciona":
            deleted = delete_relaciona(graph_conn)
            result = {"deleted_relaciona": deleted}
        elif args.phase == "reset-grafo":
            result = reset_grafo_flags(datamart_conn)
        elif args.phase == "prepare":
            deleted = delete_relaciona(graph_conn)
            reset = reset_grafo_flags(datamart_conn)
            result = {"deleted_relaciona": deleted, **reset}
        else:
            result = validate(datamart_conn, graph_conn)

        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "phase": args.phase,
            "graph_name": settings.graph_name,
            "result": result,
        }
        text = json.dumps(payload, ensure_ascii=False, indent=2)
        print(text)

        if args.output:
            out = Path(args.output)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(text + "\n", encoding="utf-8")
    finally:
        datamart_conn.close()
        graph_conn.close()


if __name__ == "__main__":
    main()
