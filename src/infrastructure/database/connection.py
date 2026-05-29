from __future__ import annotations

import psycopg
from psycopg import sql

from config.settings import settings


def get_postgres_connection(dbname: str | None = None, schema: str | None = None) -> psycopg.Connection:
    conn = psycopg.connect(
        host=settings.postgres_host,
        port=settings.postgres_port,
        dbname=dbname or settings.postgres_db,
        user=settings.postgres_user,
        password=settings.postgres_password,
        autocommit=True,
    )

    if schema:
        with conn.cursor() as cursor:
            cursor.execute(sql.SQL("SET search_path TO {}, public;").format(sql.Identifier(schema)))

    return conn


def init_apache_age(conn: psycopg.Connection) -> None:
    with conn.cursor() as cursor:
        cursor.execute("CREATE EXTENSION IF NOT EXISTS age;")
        cursor.execute("LOAD 'age';")
        cursor.execute("SET search_path = ag_catalog, '$user', public;")
        cursor.execute(
            "SELECT CASE WHEN NOT EXISTS ("
            "SELECT 1 FROM ag_catalog.ag_graph WHERE name = %s"
            ") THEN create_graph(%s) ELSE NULL END;",
            (settings.graph_name, settings.graph_name),
        )
