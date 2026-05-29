from __future__ import annotations

import argparse
import json

from src.application.pipeline_use_case import PipelineUseCase


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
    return parser


def main() -> None:
    args = build_parser().parse_args()
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
