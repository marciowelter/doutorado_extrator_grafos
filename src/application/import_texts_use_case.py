from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, TypeVar

import psycopg

from src.application.pipeline_use_case import PipelineUseCase
from src.infrastructure.database.connection import get_postgres_connection

IMPORT_SQL = """
select
  discurso.id as discurso_id,
  trecho.id as trecho_id,
  CONCAT(
    'quando: "', TO_CHAR(discurso.quando, 'DD/MM/YYYY'), '", ',
    'como: "', como.como, '", ',
    'onde: "', onde.nome, '", ',
    'porque: "', porque.porque, '", ',
    'quem: "', orador.nome, '", ',
    'oque: "', REPLACE(trecho.texto, CHR(34), CHR(39)), '" '
  ) as texto
from
  doutorado.datamart_discurso discurso
join doutorado.datamart_trecho trecho on trecho.discurso_id = discurso.id
join doutorado.datamart_orador orador on orador.id = trecho.orador_id
join doutorado.datamart_como como on como.id = discurso.como_id
join doutorado.datamart_onde onde on onde.id = discurso.onde_id
join doutorado.datamart_porque porque on porque.id = discurso.porque_id
where
  (discurso.ata_alesc_id is not null or discurso.ata_id is not null)
  and length(trecho.texto) > 1000
  and trecho.grafo is false
order by
  discurso.id,
  trecho.seq
""".strip()

MAX_DB_ATTEMPTS = 3
DB_RETRY_DELAY_SECONDS = 2


@dataclass(frozen=True)
class ImportTextRecord:
    discurso_id: int
    trecho_id: int
    texto: str


@dataclass(frozen=True)
class ImportProgress:
    total: int
    attempted: int
    successful: int
    failed: int
    record: ImportTextRecord


@dataclass(frozen=True)
class RetryEvent:
    attempt: int
    max_attempts: int
    context: str
    error: str


ProgressCallback = Callable[[ImportProgress], None]
RetryCallback = Callable[[RetryEvent], None]
T = TypeVar("T")


class ImportTextsUseCase:
    def __init__(
        self,
        pipeline: PipelineUseCase | None = None,
        conn: psycopg.Connection | None = None,
    ) -> None:
        self._managed_pipeline = pipeline is None
        self._pipeline = pipeline or PipelineUseCase(conn=get_postgres_connection(dbname="banco"))
        self._pipeline.bootstrap()
        self._conn = conn or get_postgres_connection(dbname="banco", schema="doutorado")

    def process_all(
        self,
        on_progress: ProgressCallback | None = None,
        on_retry: RetryCallback | None = None,
    ) -> dict[str, int]:
        total = self._run_with_retry(
            self.count_pending_records,
            on_retry=on_retry,
            context="contagem de registros pendentes",
        )

        if total == 0:
            return {"total": 0, "attempted": 0, "successful": 0, "failed": 0}

        records = self._run_with_retry(
            self.fetch_pending_records,
            on_retry=on_retry,
            context="carregamento de registros pendentes",
        )

        attempted = 0
        successful = 0
        failed = 0

        for record in records:
            attempted += 1
            try:
                self._run_with_retry(
                    lambda current_record=record: self._pipeline.process_text(current_record.texto),
                    on_retry=on_retry,
                    context=f"processamento do trecho_id={record.trecho_id}",
                )
                self._run_with_retry(
                    lambda current_record=record: self.mark_record_processed(current_record.trecho_id),
                    on_retry=on_retry,
                    context=f"atualizacao de status do trecho_id={record.trecho_id}",
                )
                successful += 1
            except Exception:
                failed += 1

            if on_progress is not None:
                on_progress(
                    ImportProgress(
                        total=total,
                        attempted=attempted,
                        successful=successful,
                        failed=failed,
                        record=record,
                    )
                )

        return {
            "total": total,
            "attempted": attempted,
            "successful": successful,
            "failed": failed,
        }

    def count_pending_records(self) -> int:
        count_sql = f"SELECT COUNT(*) FROM ({IMPORT_SQL}) AS pending_texts"
        with self._conn.cursor() as cursor:
            cursor.execute(count_sql)
            row = cursor.fetchone()
        return int(row[0]) if row else 0

    def fetch_pending_records(self) -> list[ImportTextRecord]:
        with self._conn.cursor() as cursor:
            cursor.execute(IMPORT_SQL)
            rows = cursor.fetchall()

        return [
            ImportTextRecord(
                discurso_id=int(discurso_id),
                trecho_id=int(trecho_id),
                texto=str(texto),
            )
            for discurso_id, trecho_id, texto in rows
        ]

    def mark_record_processed(self, trecho_id: int) -> None:
        update_sql = "UPDATE doutorado.datamart_trecho SET grafo = TRUE WHERE id = %s"
        with self._conn.cursor() as cursor:
            cursor.execute(update_sql, (trecho_id,))

    def _run_with_retry(
        self,
        operation: Callable[[], T],
        *,
        on_retry: RetryCallback | None,
        context: str,
    ) -> T:
        error: Exception | None = None

        for attempt in range(1, MAX_DB_ATTEMPTS + 1):
            try:
                return operation()
            except Exception as exc:
                error = exc
                if attempt >= MAX_DB_ATTEMPTS:
                    break
                self._reconnect_databases()
                if on_retry is not None:
                    on_retry(
                        RetryEvent(
                            attempt=attempt,
                            max_attempts=MAX_DB_ATTEMPTS,
                            context=context,
                            error=str(exc),
                        )
                    )
                time.sleep(DB_RETRY_DELAY_SECONDS)

        assert error is not None
        raise error

    def _reconnect_databases(self) -> None:
        self._conn = get_postgres_connection(dbname="banco", schema="doutorado")
        if self._managed_pipeline:
            self._pipeline = PipelineUseCase(conn=get_postgres_connection(dbname="banco"))
            self._pipeline.bootstrap()
        else:
            self._pipeline.reconnect()
