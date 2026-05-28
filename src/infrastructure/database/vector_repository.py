from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

import psycopg
from psycopg import sql
from llama_index.core import Settings
from pgvector.psycopg import register_vector

from config.settings import settings
from src.domain.repositories import VectorRepository as VectorRepositoryPort


logger = logging.getLogger(__name__)


class VectorRepository(VectorRepositoryPort):
    def __init__(self, conn: psycopg.Connection):
        self._conn = conn
        register_vector(self._conn)

    def ensure_store(self) -> None:
        target_dim = self._resolve_target_dim()
        current_dim = self._get_table_vector_dim()

        with self._conn.cursor() as cursor:
            cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")

        if current_dim is None:
            self._create_vector_table(target_dim)
            return

        if current_dim != target_dim:
            self._migrate_vector_table(current_dim=current_dim, target_dim=target_dim)

    def upsert_chunk(self, chunk_text: str, metadata: dict[str, str] | None = None) -> None:
        embed_model = Settings.embed_model
        if embed_model is None:
            raise RuntimeError("Settings.embed_model is not configured")

        vector = embed_model.get_text_embedding(chunk_text)
        table_dim = self._get_table_vector_dim()
        if table_dim is not None and len(vector) != table_dim:
            logger.warning(
                "Skipping vector insert due to embedding dimension mismatch. got=%s table_dim=%s table=%s",
                len(vector),
                table_dim,
                settings.vector_table,
            )
            return

        payload = json.dumps(metadata or {})
        with self._conn.cursor() as cursor:
            cursor.execute(
                f"""
                INSERT INTO {settings.vector_table} (content, embedding, metadata)
                VALUES (%s, %s::vector, %s::jsonb);
                """,
                (chunk_text, vector, payload),
            )

    def similarity_search(self, query: str, limit: int = 5) -> list[dict[str, str]]:
        embed_model = Settings.embed_model
        if embed_model is None:
            raise RuntimeError("Settings.embed_model is not configured")

        query_vector = embed_model.get_text_embedding(query)
        with self._conn.cursor() as cursor:
            cursor.execute(
                f"""
                SELECT content, metadata::text
                FROM {settings.vector_table}
                ORDER BY embedding <=> %s::vector
                LIMIT %s;
                """,
                (query_vector, limit),
            )
            rows = cursor.fetchall()

        return [{"content": content, "metadata": metadata} for content, metadata in rows]

    def _resolve_target_dim(self) -> int:
        embed_model = Settings.embed_model
        if embed_model is None:
            return settings.vector_dim
        try:
            probe = embed_model.get_text_embedding("dim_probe")
            return len(probe)
        except Exception:
            return settings.vector_dim

    def _get_table_vector_dim(self) -> int | None:
        with self._conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT a.atttypmod
                FROM pg_attribute a
                WHERE a.attrelid = to_regclass(%s)
                  AND a.attname = 'embedding'
                  AND a.attnum > 0
                LIMIT 1;
                """,
                (settings.vector_table,),
            )
            row = cursor.fetchone()

        if not row:
            return None

        atttypmod = int(row[0])
        if atttypmod <= 0:
            return None
        return atttypmod

    def _create_vector_table(self, dim: int) -> None:
        table_ident = self._table_identifier()
        query = sql.SQL(
            """
            CREATE TABLE IF NOT EXISTS {} (
                id BIGSERIAL PRIMARY KEY,
                content TEXT NOT NULL,
                embedding vector({}) NOT NULL,
                metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );
            """
        ).format(table_ident, sql.SQL(str(dim)))

        with self._conn.cursor() as cursor:
            cursor.execute(query)

    def _migrate_vector_table(self, current_dim: int, target_dim: int) -> None:
        schema_name, table_name = self._split_table_name()
        suffix = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
        backup_table_name = f"{table_name}_bak_{current_dim}d_{suffix}"

        source_ident = self._table_identifier()

        with self._conn.cursor() as cursor:
            cursor.execute(sql.SQL("ALTER TABLE {} RENAME TO {};").format(source_ident, sql.Identifier(backup_table_name)))

        self._create_vector_table(target_dim)

        logger.warning(
            "Vector table dimension mismatch detected. old_dim=%s new_dim=%s old_table_backed_up_as=%s",
            current_dim,
            target_dim,
            f"{schema_name}.{backup_table_name}" if schema_name else backup_table_name,
        )

    def _split_table_name(self) -> tuple[str | None, str]:
        if "." in settings.vector_table:
            schema_name, table_name = settings.vector_table.split(".", 1)
            return schema_name, table_name
        return None, settings.vector_table

    def _table_identifier(self) -> sql.SQL | sql.Identifier:
        schema_name, table_name = self._split_table_name()
        if schema_name:
            return sql.SQL("{}.{}").format(sql.Identifier(schema_name), sql.Identifier(table_name))
        return sql.Identifier(table_name)
