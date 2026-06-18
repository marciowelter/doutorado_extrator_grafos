from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path

from src.application.correct_labels_use_case import CorrectLabelsUseCase, LabelCorrectionRecord
from src.application.import_texts_use_case import (
    ImportTextsUseCase,
    ImportProgress,
    RecordFailureEvent,
    RetryEvent,
)
from src.application.pipeline_use_case import PipelineUseCase


DEFAULT_BACKGROUND_LOG = "import_texts_background.log"
DEFAULT_PROGRESS_BATCH_SIZE = 25
DEFAULT_LABEL_CORRECTION_LOG = "/tmp/doutorado_label_correction.log"
DEFAULT_LABEL_CORRECTION_LIMIT = 100
DEFAULT_LABEL_CORRECTION_PROVIDER = "gemini"


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
        help="Executa o job em background (com --import-texts ou --correct-labels)",
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
    parser.add_argument(
        "--correct-labels",
        action="store_true",
        help="Classifica labels TEMA/ENTIDADE via LLM e corrige registros quando necessario",
    )
    parser.add_argument(
        "--correct-labels-provider",
        type=str,
        choices=("ollama", "gemini"),
        default=DEFAULT_LABEL_CORRECTION_PROVIDER,
        help="Provedor LLM usado no modo --correct-labels (ollama ou gemini)",
    )
    parser.add_argument(
        "--correct-labels-limit",
        type=int,
        default=DEFAULT_LABEL_CORRECTION_LIMIT,
        help="Quantidade de nos analisados no modo --correct-labels",
    )
    parser.add_argument(
        "--correct-labels-offset",
        type=int,
        default=0,
        help="Deslocamento inicial de nos no modo --correct-labels",
    )
    parser.add_argument(
        "--correct-labels-log",
        type=str,
        default=DEFAULT_LABEL_CORRECTION_LOG,
        help="Arquivo temporario de log gerado no modo --correct-labels",
    )
    parser.add_argument(
        "--correct-labels-dry-run",
        action="store_true",
        help="Somente analisa e registra no log, sem atualizar o grafo",
    )
    parser.add_argument(
        "--correct-labels-all",
        action="store_true",
        help="Processa todos os nos Entidade do grafo (ignora --correct-labels-limit e --correct-labels-offset)",
    )
    parser.add_argument(
        "--correct-labels-random",
        action="store_true",
        help="Seleciona nos aleatorios (incompativel com --correct-labels-all)",
    )
    parser.add_argument(
        "--correct-labels-exclude-first",
        type=int,
        default=0,
        help="Exclui os N primeiros nos (por id) ao usar --correct-labels-random",
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
                "event": "retry",
                "retry_scope": "database",
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


def _run_correct_labels(
    *,
    limit: int | None,
    offset: int,
    log_path: str,
    dry_run: bool,
    provider: str,
    random_sample: bool,
    exclude_first: int,
) -> dict[str, int | str | bool]:
    use_case = CorrectLabelsUseCase(provider=provider)

    def on_progress(record: LabelCorrectionRecord, current: int, total: int) -> None:
        if record.error:
            status = "error"
        elif record.updated:
            status = "updated"
        elif record.would_update:
            status = "would_update"
        else:
            status = "unchanged"
        _print_json_log(
            {
                "timestamp": _now_iso(),
                "event": "label_correction_progress",
                "current": current,
                "total": total,
                "status": status,
                "provider": provider,
                "node_id": record.node_id,
                "name": record.name,
                "current_label": record.current_label,
                "predicted_label": record.predicted_label,
            }
        )

    summary = use_case.process_batch(
        limit=limit,
        offset=max(0, int(offset)),
        random_sample=random_sample,
        exclude_first=max(0, int(exclude_first)),
        log_path=log_path,
        dry_run=dry_run,
        on_progress=on_progress,
    )
    return {
        "processed": summary.processed,
        "updated": summary.updated,
        "would_update": summary.would_update,
        "unchanged": summary.unchanged,
        "errors": summary.errors,
        "log_path": summary.log_path,
        "dry_run": dry_run,
        "provider": provider,
        "random_sample": random_sample,
        "exclude_first": max(0, int(exclude_first)),
        "process_entire_graph": limit is None and not random_sample,
    }


def _spawn_background_correct_labels(
    *,
    log_path: str,
    provider: str,
    limit: int | None,
    offset: int,
    dry_run: bool,
    random_sample: bool,
    exclude_first: int,
) -> int:
    target_log = Path(log_path).resolve()
    target_log.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "main.py",
        "--correct-labels",
        "--correct-labels-provider",
        provider,
        "--correct-labels-offset",
        str(max(0, int(offset))),
        "--correct-labels-log",
        str(target_log),
        "--correct-labels-exclude-first",
        str(max(0, int(exclude_first))),
    ]
    if limit is None:
        cmd.append("--correct-labels-all")
    else:
        cmd.extend(["--correct-labels-limit", str(max(1, int(limit)))])
    if dry_run:
        cmd.append("--correct-labels-dry-run")
    if random_sample:
        cmd.append("--correct-labels-random")

    with target_log.open("a", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            cwd=str(Path(__file__).resolve().parent),
        )

    return process.pid


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


def _validate_batch_args(args: argparse.Namespace) -> None:
    if args.background and not args.import_texts and not args.correct_labels:
        raise SystemExit("Use --background somente junto com --import-texts ou --correct-labels.")

    if args.correct_labels_all and args.correct_labels_random:
        raise SystemExit("Use --correct-labels-all ou --correct-labels-random, nao ambos.")

    if args.correct_labels_all and args.correct_labels_offset:
        raise SystemExit("--correct-labels-offset nao se aplica com --correct-labels-all.")


def main() -> None:
    args = build_parser().parse_args()
    _validate_batch_args(args)

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

    if args.correct_labels:
        selected_limit = None if args.correct_labels_all else args.correct_labels_limit

        if args.background:
            pid = _spawn_background_correct_labels(
                log_path=args.correct_labels_log,
                provider=args.correct_labels_provider,
                limit=selected_limit,
                offset=args.correct_labels_offset,
                dry_run=args.correct_labels_dry_run,
                random_sample=args.correct_labels_random,
                exclude_first=args.correct_labels_exclude_first,
            )
            payload = {
                "status": "started",
                "job": "correct_labels",
                "pid": pid,
                "log": str(Path(args.correct_labels_log).resolve()),
                "provider": args.correct_labels_provider,
                "process_entire_graph": args.correct_labels_all,
                "dry_run": args.correct_labels_dry_run,
                "hint": "Acompanhe com: tail -f <log>",
            }
            print(json.dumps(payload, ensure_ascii=False, indent=2))
            return

        summary = _run_correct_labels(
            limit=selected_limit,
            offset=args.correct_labels_offset,
            log_path=args.correct_labels_log,
            dry_run=args.correct_labels_dry_run,
            provider=args.correct_labels_provider,
            random_sample=args.correct_labels_random,
            exclude_first=args.correct_labels_exclude_first,
        )
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
