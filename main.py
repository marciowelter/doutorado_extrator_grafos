from __future__ import annotations

import argparse
import json

from src.application.pipeline_use_case import PipelineUseCase


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LlamaIndex KG pipeline")
    parser.add_argument("--limit", type=int, default=None, help="Chunk processing limit")
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="Single experimental text to process instead of source table",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    pipeline = PipelineUseCase()
    pipeline.bootstrap()

    if args.text:
        extraction = pipeline.process_text(args.text)
        print(json.dumps(extraction.model_dump(), ensure_ascii=False, indent=2))
        return

    processed = pipeline.process_source_chunks(limit=args.limit)
    print(f"Processed chunks: {processed}")


if __name__ == "__main__":
    main()
