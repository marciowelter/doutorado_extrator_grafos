from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from src.application.import_texts_use_case import ImportTextsUseCase
from src.application.pipeline_use_case import PipelineUseCase


DEFAULT_BACKGROUND_LOG = "import_texts_background.log"


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
    return parser


def _run_import_texts() -> dict[str, int | float]:
    use_case = ImportTextsUseCase()
    return use_case.process_all()


def _spawn_background_import(log_path: str) -> int:
    target_log = Path(log_path).resolve()
    target_log.parent.mkdir(parents=True, exist_ok=True)

    cmd = [sys.executable, "main.py", "--import-texts"]
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
        pid = _spawn_background_import(args.background_log)
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
        summary = _run_import_texts()
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
