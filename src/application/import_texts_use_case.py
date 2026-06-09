from __future__ import annotations

import time
import traceback
from dataclasses import dataclass
from typing import Callable, TypeVar

import psycopg

from config.settings import settings
from src.domain.models import KnowledgeGraphExtraction
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
      length(trecho.texto) > 1000
  and trecho.grafo is false
order by
  discurso.id,
  trecho.seq
""".strip()

MAX_DB_ATTEMPTS = max(1, settings.postgres_retry_attempts)
DB_RETRY_BASE_DELAY_SECONDS = max(0.0, settings.postgres_retry_base_delay_seconds)
DB_RETRY_MAX_DELAY_SECONDS = max(DB_RETRY_BASE_DELAY_SECONDS, settings.postgres_retry_max_delay_seconds)


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
    record_duration_seconds: float
    average_duration_seconds: float
    estimated_remaining_seconds: float
    estimated_completion_seconds: float


@dataclass(frozen=True)
class RetryEvent:
    attempt: int
    max_attempts: int
    context: str
    error: str


@dataclass(frozen=True)
class RecordFailureEvent:
    stage: str
    record: ImportTextRecord
    error: str
    traceback: str


ProgressCallback = Callable[[ImportProgress], None]
RetryCallback = Callable[[RetryEvent], None]
RecordFailureCallback = Callable[[RecordFailureEvent], None]
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
        on_record_failure: RecordFailureCallback | None = None,
    ) -> dict[str, int | float]:
        started_at = time.perf_counter()
        total = self._run_with_retry(
            self.count_pending_records,
            on_retry=on_retry,
            context="contagem de registros pendentes",
        )

        if total == 0:
            return {
                "total": 0,
                "attempted": 0,
                "successful": 0,
                "failed": 0,
                "elapsed_seconds": 0.0,
                "average_record_seconds": 0.0,
                "estimated_remaining_seconds": 0.0,
                "estimated_completion_seconds": 0.0,
            }

        records = self._run_with_retry(
            self.fetch_pending_records,
            on_retry=on_retry,
            context="carregamento de registros pendentes",
        )

        attempted = 0
        successful = 0
        failed = 0
        extraction_cache: dict[int, KnowledgeGraphExtraction] = {}
        themes_cache_by_discurso: dict[int, list[str]] = {}
        accumulated_record_seconds = 0.0

        for record in records:
            record_started_at = time.perf_counter()
            attempted += 1
            stage = "init"
            try:
                if record.discurso_id in themes_cache_by_discurso:
                    additional_themes = themes_cache_by_discurso[record.discurso_id]
                else:
                    stage = "fetch_datamart_themes"
                    additional_themes = self._run_with_retry(
                        lambda current_record=record: self.fetch_datamart_oque_themes(current_record.discurso_id),
                        on_retry=on_retry,
                        context=f"consulta de temas datamart_oque para discurso_id={record.discurso_id}",
                    )
                    themes_cache_by_discurso[record.discurso_id] = additional_themes

                stage = "extract_text"
                extraction_cache[record.trecho_id] = self._pipeline.extract_text(
                    record.texto,
                    additional_themes=additional_themes,
                )

                stage = "save_extraction"
                self._run_with_retry(
                    lambda current_record=record: self._pipeline.save_extraction(
                        extraction_cache[current_record.trecho_id],
                    ),
                    on_retry=on_retry,
                    context=f"persistencia do trecho_id={record.trecho_id}",
                )
                stage = "mark_record_processed"
                self._run_with_retry(
                    lambda current_record=record: self.mark_record_processed(current_record.trecho_id),
                    on_retry=on_retry,
                    context=f"atualizacao de status do trecho_id={record.trecho_id}",
                )

                extraction_cache.pop(record.trecho_id, None)
                successful += 1
            except Exception as exc:
                extraction_cache.pop(record.trecho_id, None)
                failed += 1
                if on_record_failure is not None:
                    on_record_failure(
                        RecordFailureEvent(
                            stage=stage,
                            record=record,
                            error=str(exc),
                            traceback=traceback.format_exc(),
                        )
                    )

            record_duration_seconds = round(time.perf_counter() - record_started_at, 4)
            accumulated_record_seconds += record_duration_seconds
            average_duration_seconds = accumulated_record_seconds / attempted
            estimated_remaining_seconds = average_duration_seconds * max(total - attempted, 0)
            estimated_completion_seconds = round(time.perf_counter() - started_at + estimated_remaining_seconds, 4)

            if on_progress is not None:
                on_progress(
                    ImportProgress(
                        total=total,
                        attempted=attempted,
                        successful=successful,
                        failed=failed,
                        record=record,
                        record_duration_seconds=record_duration_seconds,
                        average_duration_seconds=round(average_duration_seconds, 4),
                        estimated_remaining_seconds=round(estimated_remaining_seconds, 4),
                        estimated_completion_seconds=estimated_completion_seconds,
                    )
                )

        return {
            "total": total,
            "attempted": attempted,
            "successful": successful,
            "failed": failed,
            "elapsed_seconds": round(time.perf_counter() - started_at, 4),
            "average_record_seconds": round(accumulated_record_seconds / attempted, 4) if attempted else 0.0,
            "estimated_remaining_seconds": 0.0,
            "estimated_completion_seconds": round(time.perf_counter() - started_at, 4),
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

    def fetch_datamart_oque_themes(self, discurso_id: int) -> list[str]:
        query = (
            "SELECT tema "
            "FROM doutorado.datamart_oque "
            "WHERE discurso_id = %s AND coalesce(trim(tema), '') <> ''"
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
                if not self._is_transient_connection_error(exc):
                    raise
                if attempt >= MAX_DB_ATTEMPTS:
                    break
                try:
                    self._reconnect_databases()
                except Exception as reconnect_exc:
                    raise RuntimeError(
                        (
                            "Falha ao reconectar banco durante operacao transiente "
                            f"({context}). Erro original: {exc!r}"
                        )
                    ) from reconnect_exc
                if on_retry is not None:
                    on_retry(
                        RetryEvent(
                            attempt=attempt,
                            max_attempts=MAX_DB_ATTEMPTS,
                            context=context,
                            error=str(exc),
                        )
                    )
                backoff_seconds = min(
                    DB_RETRY_BASE_DELAY_SECONDS * (2 ** (attempt - 1)),
                    DB_RETRY_MAX_DELAY_SECONDS,
                )
                time.sleep(backoff_seconds)

        assert error is not None
        raise error

    def _is_transient_connection_error(self, error: Exception) -> bool:
        if isinstance(error, (psycopg.OperationalError, psycopg.InterfaceError)):
            return True

        sqlstate = getattr(error, "sqlstate", None)
        if isinstance(sqlstate, str) and sqlstate.startswith("08"):
            return True

        lowered = str(error).lower()
        transient_hints = (
            "consuming input failed",
            "ssl error",
            "unexpected eof",
            "connection reset",
            "server closed the connection unexpectedly",
            "connection not open",
            "broken pipe",
        )
        if any(hint in lowered for hint in transient_hints):
            return True

        return False

    def _reconnect_databases(self) -> None:
        old_conn = self._conn
        self._conn = get_postgres_connection(dbname="banco", schema="doutorado")
        try:
            old_conn.close()
        except Exception:
            pass

        if self._managed_pipeline:
            old_pipeline_conn = self._pipeline._conn
            self._pipeline = PipelineUseCase(conn=get_postgres_connection(dbname="banco"))
            self._pipeline.bootstrap()
            try:
                old_pipeline_conn.close()
            except Exception:
                pass
        else:
            self._pipeline.reconnect()
