from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path

from src.application.import_texts_use_case import (
    ImportTextsUseCase,
    ImportProgress,
    RecordFailureEvent,
    RetryEvent,
)
from src.application.pipeline_use_case import PipelineUseCase


DEFAULT_BACKGROUND_LOG = "import_texts_background.log"
DEFAULT_PROGRESS_BATCH_SIZE = 25


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Doutorado Extrator Grafos pipeline")
    parser.add_argument(
        "--text",
        type=str,
        required=False,
        help="Texto de entrada para extracao de entidades/temas e persistencia no grafo",
    )
    parser.add_argument(
        "--normalize-graph",
        action="store_true",
        help="Normaliza e unifica entidades/temas no grafo existente",
    )
    parser.add_argument(
        "--import-texts",
        action="store_true",
        help="Importa textos pendentes e processa no grafo",
    )
    parser.add_argument(
        "--background",
        action="store_true",
        help="Executa importacao em background (somente com --import-texts)",
    )
    parser.add_argument(
        "--background-log",
        type=str,
        default=DEFAULT_BACKGROUND_LOG,
        help="Arquivo de log usado na execucao em background",
    )
    parser.add_argument(
        "--progress-batch-size",
        type=int,
        default=DEFAULT_PROGRESS_BATCH_SIZE,
        help="Quantidade de registros por lote para log de progresso no modo --import-texts",
    )
    return parser


def _now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _print_json_log(payload: dict) -> None:
    print(json.dumps(payload, ensure_ascii=False), flush=True)


def _run_import_texts(progress_batch_size: int) -> dict[str, int | float]:
    use_case = ImportTextsUseCase()
    last_logged_batch = 0
    safe_batch_size = max(1, int(progress_batch_size or DEFAULT_PROGRESS_BATCH_SIZE))

    def on_retry(event: RetryEvent) -> None:
        _print_json_log(
            {
                "timestamp": _now_iso(),
                "event": "db_retry",
                "attempt": event.attempt,
                "max_attempts": event.max_attempts,
                "context": event.context,
                "error": event.error,
            }
        )

    def on_record_failure(event: RecordFailureEvent) -> None:
        _print_json_log(
            {
                "timestamp": _now_iso(),
                "event": "record_failure",
                "stage": event.stage,
                "discurso_id": event.record.discurso_id,
                "trecho_id": event.record.trecho_id,
                "error": event.error,
                "traceback": event.traceback,
            }
        )

    def on_progress(event: ImportProgress) -> None:
        nonlocal last_logged_batch
        current_batch = (event.attempted - 1) // safe_batch_size + 1
        should_log = event.attempted == event.total or current_batch > last_logged_batch
        if not should_log:
            return

        last_logged_batch = current_batch
        _print_json_log(
            {
                "timestamp": _now_iso(),
                "event": "batch_progress",
                "batch": current_batch,
                "batch_size": safe_batch_size,
                "attempted": event.attempted,
                "total": event.total,
                "successful": event.successful,
                "failed": event.failed,
                "current_discurso_id": event.record.discurso_id,
                "current_trecho_id": event.record.trecho_id,
                "average_record_seconds": event.average_duration_seconds,
                "estimated_remaining_seconds": event.estimated_remaining_seconds,
            }
        )

    return use_case.process_all(
        on_progress=on_progress,
        on_retry=on_retry,
        on_record_failure=on_record_failure,
    )


def _spawn_background_import(log_path: str, progress_batch_size: int) -> int:
    target_log = Path(log_path).resolve()
    target_log.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "main.py",
        "--import-texts",
        "--progress-batch-size",
        str(max(1, int(progress_batch_size))),
    ]
    with target_log.open("a", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            cwd=str(Path(__file__).resolve().parent),
        )

    return process.pid


def main() -> None:
    args = build_parser().parse_args()

    if args.background and not args.import_texts:
        raise SystemExit("Use --background somente junto com --import-texts.")

    if args.import_texts and args.background:
        pid = _spawn_background_import(args.background_log, args.progress_batch_size)
        payload = {
            "status": "started",
            "job": "import_texts",
            "pid": pid,
            "log": str(Path(args.background_log).resolve()),
            "hint": "Acompanhe com: tail -f <log>",
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    if args.import_texts:
        summary = _run_import_texts(args.progress_batch_size)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    pipeline = PipelineUseCase()
    pipeline.bootstrap()

    if args.normalize_graph:
        report = pipeline.normalize_and_unify_graph()
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return

    if not args.text:
        raise SystemExit("Informe --text para extracao ou use --normalize-graph para revisar o grafo.")

    extraction = pipeline.process_text(args.text)
    print(json.dumps(extraction.model_dump(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
