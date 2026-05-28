from __future__ import annotations

import psycopg

from config.settings import settings
from src.domain.repositories import ChunkRepository


class PostgresChunkRepository(ChunkRepository):
    def __init__(self, conn: psycopg.Connection) -> None:
        self._conn = conn

    def fetch_chunks(self, limit: int) -> list[str]:
        with self._conn.cursor() as cursor:
            cursor.execute(
                f"""
                SELECT {settings.source_text_column}
                FROM {settings.postgres_schema}.{settings.source_table}
                WHERE {settings.source_text_column} IS NOT NULL
                LIMIT %s;
                """,
                (limit,),
            )
            rows = cursor.fetchall()
        return [row[0] for row in rows if row and row[0]]
