from __future__ import annotations

import argparse
import json

from src.application.pipeline_use_case import PipelineUseCase


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Doutorado Extrator Grafos pipeline")
    parser.add_argument(
        "--text",
        type=str,
        required=True,
        help="Texto de entrada para extracao de entidades/temas e persistencia no grafo",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    pipeline = PipelineUseCase()
    pipeline.bootstrap()

    extraction = pipeline.process_text(args.text)
    print(json.dumps(extraction.model_dump(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
