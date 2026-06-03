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
        connect_timeout=settings.postgres_connect_timeout,
        keepalives=1,
        keepalives_idle=settings.postgres_keepalives_idle,
        keepalives_interval=settings.postgres_keepalives_interval,
        keepalives_count=settings.postgres_keepalives_count,
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
        _ensure_entity_name_index(cursor)


def _ensure_entity_name_index(cursor: psycopg.Cursor) -> None:
    # Tenta a sintaxe nativa do AGE primeiro; em ambientes sem suporte, cai para
    # índice por expressão na tabela física do label.
    statements = [
        sql.SQL(
            "SELECT * FROM cypher({}, $$ "
            "CREATE INDEX IF NOT EXISTS FOR (n:Entidade) ON (n.name) "
            "$$) as (result agtype);"
        ).format(sql.Literal(settings.graph_name)),
        sql.SQL(
            "CREATE INDEX IF NOT EXISTS {} ON {}.{} ((properties ->> 'name'));"
        ).format(
            sql.Identifier(f"idx_{settings.graph_name}_entidade_name"),
            sql.Identifier(settings.graph_name),
            sql.Identifier("Entidade"),
        ),
    ]

    for statement in statements:
        try:
            cursor.execute(statement)
            return
        except psycopg.Error as exc:
            # O label pode ainda nao existir (42P01) no primeiro bootstrap.
            # Nesse caso, apenas segue para nao bloquear a inicializacao.
            if exc.sqlstate in {"42P07", "42P01"}:
                return
            continue
