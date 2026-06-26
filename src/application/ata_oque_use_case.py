from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

import psycopg

from src.infrastructure.database.connection import get_postgres_connection
from src.infrastructure.llm.ata_theme_extractor import AtaThemeExtractor, MAX_ATA_TEMAS


ATA_DISCURSO_WITHOUT_OQUE_SQL = """
SELECT d.id
FROM doutorado.datamart_discurso d
WHERE (d.ata_id IS NOT NULL OR d.ata_alesc_id IS NOT NULL)
  AND NOT EXISTS (
    SELECT 1
    FROM doutorado.datamart_oque o
    WHERE o.discurso_id = d.id
      AND coalesce(trim(o.tema), '') <> ''
  )
ORDER BY d.id
""".strip()

IS_ATA_DISCURSO_SQL = """
SELECT 1
FROM doutorado.datamart_discurso d
WHERE d.id = %s
  AND (d.ata_id IS NOT NULL OR d.ata_alesc_id IS NOT NULL)
LIMIT 1
""".strip()

DISCURSO_CONTEXT_SQL = """
SELECT
  coalesce(a_sp.titulo_ata, a_alesc.ementa, a_alesc.tipo_evento, '') AS titulo,
  coalesce(como.como, '') AS como,
  coalesce(porque.porque, '') AS porque,
  coalesce(string_agg(t.texto, E'\\n\\n' ORDER BY t.seq), '') AS texto
FROM doutorado.datamart_discurso d
JOIN doutorado.datamart_como como ON como.id = d.como_id
JOIN doutorado.datamart_porque porque ON porque.id = d.porque_id
LEFT JOIN doutorado.datamart_trecho t ON t.discurso_id = d.id
LEFT JOIN doutorado.atas_sessoes_plenarias_alesc a_sp ON a_sp.id = d.ata_id
LEFT JOIN doutorado.atas_alesc a_alesc ON a_alesc.id = d.ata_alesc_id
WHERE d.id = %s
GROUP BY d.id, a_sp.titulo_ata, a_alesc.ementa, a_alesc.tipo_evento, como.como, porque.porque
""".strip()

INSERT_OQUE_SQL = """
INSERT INTO doutorado.datamart_oque (seq, tema, discurso_id, updated_at)
VALUES (%s, %s, %s, NOW())
""".strip()

DELETE_OQUE_SQL = "DELETE FROM doutorado.datamart_oque WHERE discurso_id = %s"

RESET_ATA_GRAFO_SQL = """
UPDATE doutorado.datamart_trecho t
SET grafo = FALSE
FROM doutorado.datamart_discurso d
WHERE t.discurso_id = d.id
  AND (d.ata_id IS NOT NULL OR d.ata_alesc_id IS NOT NULL)
""".strip()


@dataclass(frozen=True)
class AtaOqueProgress:
    total: int
    processed: int
    populated: int
    skipped: int
    failed: int
    current_discurso_id: int | None


@dataclass(frozen=True)
class AtaOqueSummary:
    total: int
    populated: int
    skipped: int
    failed: int
    trechos_reset: int
    elapsed_seconds: float


ProgressCallback = Callable[[AtaOqueProgress], None]


class AtaOqueUseCase:
    def __init__(
        self,
        conn: psycopg.Connection | None = None,
        theme_extractor: AtaThemeExtractor | None = None,
    ) -> None:
        self._conn = conn or get_postgres_connection(dbname="banco", schema="doutorado")
        self._theme_extractor = theme_extractor or AtaThemeExtractor()

    def is_ata_discurso(self, discurso_id: int) -> bool:
        with self._conn.cursor() as cursor:
            cursor.execute(IS_ATA_DISCURSO_SQL, (discurso_id,))
            return cursor.fetchone() is not None

    def load_existing_themes(self, discurso_id: int) -> list[str]:
        query = (
            "SELECT tema "
            "FROM doutorado.datamart_oque "
            "WHERE discurso_id = %s AND coalesce(trim(tema), '') <> '' "
            "ORDER BY seq"
        )
        with self._conn.cursor() as cursor:
            cursor.execute(query, (discurso_id,))
            rows = cursor.fetchall()

        themes_by_name: dict[str, None] = {}
        for (raw_theme,) in rows:
            normalized = str(raw_theme).strip().upper()
            if not normalized:
                continue
            themes_by_name[normalized] = None
        return list(themes_by_name.keys())

    def ensure_themes_for_discurso(self, discurso_id: int) -> list[str]:
        existing = self.load_existing_themes(discurso_id)
        if existing:
            return existing

        if not self.is_ata_discurso(discurso_id):
            return []

        themes = self._extract_themes_for_discurso(discurso_id)
        if not themes:
            return []

        self._persist_themes(discurso_id, themes)
        return [theme.strip().upper() for theme in themes if theme.strip()]

    def count_pending(self) -> int:
        with self._conn.cursor() as cursor:
            cursor.execute(f"SELECT COUNT(*) FROM ({ATA_DISCURSO_WITHOUT_OQUE_SQL}) pending")
            row = cursor.fetchone()
        return int(row[0]) if row else 0

    def populate_all(
        self,
        *,
        limit: int | None = None,
        reset_ata_grafo: bool = False,
        on_progress: ProgressCallback | None = None,
    ) -> AtaOqueSummary:
        started_at = time.perf_counter()
        discurso_ids = self._fetch_pending_discurso_ids(limit=limit)
        total = len(discurso_ids)

        populated = 0
        skipped = 0
        failed = 0
        trechos_reset = 0

        if reset_ata_grafo:
            trechos_reset = self.reset_ata_grafo_flags()

        for index, discurso_id in enumerate(discurso_ids, start=1):
            try:
                themes = self.ensure_themes_for_discurso(discurso_id)
                if themes:
                    populated += 1
                else:
                    skipped += 1
            except Exception:
                failed += 1

            if on_progress is not None:
                on_progress(
                    AtaOqueProgress(
                        total=total,
                        processed=index,
                        populated=populated,
                        skipped=skipped,
                        failed=failed,
                        current_discurso_id=discurso_id,
                    )
                )

        return AtaOqueSummary(
            total=total,
            populated=populated,
            skipped=skipped,
            failed=failed,
            trechos_reset=trechos_reset,
            elapsed_seconds=round(time.perf_counter() - started_at, 4),
        )

    def reset_ata_grafo_flags(self) -> int:
        with self._conn.cursor() as cursor:
            cursor.execute(RESET_ATA_GRAFO_SQL)
            updated = cursor.rowcount
        self._conn.commit()
        return int(updated)

    def _fetch_pending_discurso_ids(self, limit: int | None = None) -> list[int]:
        query = ATA_DISCURSO_WITHOUT_OQUE_SQL
        params: tuple[()] | tuple[int] = ()
        if limit is not None:
            query = f"{query}\nLIMIT %s"
            params = (max(1, int(limit)),)

        with self._conn.cursor() as cursor:
            cursor.execute(query, params)
            rows = cursor.fetchall()
        return [int(row[0]) for row in rows]

    def _extract_themes_for_discurso(self, discurso_id: int) -> list[str]:
        with self._conn.cursor() as cursor:
            cursor.execute(DISCURSO_CONTEXT_SQL, (discurso_id,))
            row = cursor.fetchone()

        if not row:
            return []

        titulo, como, porque, texto = row
        return self._theme_extractor.extract_themes(
            titulo=str(titulo or ""),
            como=str(como or ""),
            porque=str(porque or ""),
            texto=str(texto or ""),
        )

    def _persist_themes(self, discurso_id: int, themes: list[str]) -> None:
        trimmed = [theme.strip()[:200] for theme in themes if theme.strip()][:MAX_ATA_TEMAS]
        if not trimmed:
            return

        with self._conn.cursor() as cursor:
            cursor.execute(DELETE_OQUE_SQL, (discurso_id,))
            for seq, tema in enumerate(trimmed, start=1):
                cursor.execute(INSERT_OQUE_SQL, (seq, tema, discurso_id))
        self._conn.commit()
