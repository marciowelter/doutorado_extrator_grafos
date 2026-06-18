from __future__ import annotations

import datetime as dt
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import psycopg

from src.domain.normalization import normalize_graph_category
from src.infrastructure.database.age_repository import AgeRepository, _GraphNode
from src.infrastructure.database.connection import get_postgres_connection, init_apache_age
from src.infrastructure.llm.label_classifier import LabelClassifier, create_label_classifier


@dataclass(frozen=True)
class LabelCorrectionRecord:
    node_id: int
    name: str
    current_label: str
    predicted_label: str
    categoria: str
    contexto: str
    would_update: bool
    updated: bool
    justificativa: str
    error: str | None = None


@dataclass(frozen=True)
class LabelCorrectionSummary:
    processed: int
    updated: int
    would_update: int
    unchanged: int
    errors: int
    log_path: str


class CorrectLabelsUseCase:
    def __init__(
        self,
        conn: psycopg.Connection | None = None,
        graph_repo: AgeRepository | None = None,
        classifier: LabelClassifier | None = None,
        provider: str = "ollama",
    ) -> None:
        self._conn = conn or get_postgres_connection()
        init_apache_age(self._conn)
        self._graph_repo = graph_repo or AgeRepository(self._conn)
        self._provider = provider.strip().lower()
        self._classifier = classifier or create_label_classifier(self._provider)

    def process_batch(
        self,
        *,
        limit: int | None = 100,
        offset: int = 0,
        random_sample: bool = False,
        exclude_first: int = 0,
        log_path: str | Path = "/tmp/doutorado_label_correction.log",
        dry_run: bool = False,
        on_progress: Callable[[LabelCorrectionRecord, int, int], None] | None = None,
    ) -> LabelCorrectionSummary:
        safe_offset = max(0, int(offset))
        safe_exclude_first = max(0, int(exclude_first))
        if limit is None:
            safe_limit: int | None = None
        else:
            safe_limit = max(1, int(limit))
        target_log = Path(log_path).resolve()
        target_log.parent.mkdir(parents=True, exist_ok=True)

        if random_sample:
            if safe_limit is None:
                raise ValueError("Amostragem aleatoria exige --correct-labels-limit")
            nodes = self._graph_repo.fetch_random_entities(
                safe_limit,
                exclude_first=safe_exclude_first,
            )
        else:
            nodes = self._graph_repo.fetch_entities(limit=safe_limit, offset=safe_offset)
        total = len(nodes)
        total_entities_in_graph = self._graph_repo.count_entities()

        updated_count = 0
        would_update_count = 0
        unchanged_count = 0
        error_count = 0

        with target_log.open("w", encoding="utf-8") as log_file:
            log_file.write(
                json.dumps(
                    {
                        "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
                        "event": "batch_started",
                        "provider": self._provider,
                        "limit": safe_limit,
                        "offset": safe_offset,
                        "random_sample": random_sample,
                        "exclude_first": safe_exclude_first,
                        "process_entire_graph": safe_limit is None and not random_sample,
                        "total_entities_in_graph": total_entities_in_graph,
                        "dry_run": dry_run,
                        "total_nodes": total,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

            for index, node in enumerate(nodes, start=1):
                record = self._process_node(node, dry_run=dry_run)
                log_file.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
                log_file.flush()

                if record.error:
                    error_count += 1
                elif record.updated:
                    updated_count += 1
                elif record.would_update:
                    would_update_count += 1
                else:
                    unchanged_count += 1

                if on_progress is not None:
                    on_progress(record, index, total)

            summary = LabelCorrectionSummary(
                processed=total,
                updated=updated_count,
                would_update=would_update_count,
                unchanged=unchanged_count,
                errors=error_count,
                log_path=str(target_log),
            )
            log_file.write(
                json.dumps(
                    {
                        "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
                        "event": "batch_finished",
                        **asdict(summary),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

        return summary

    def _process_node(self, node: _GraphNode, *, dry_run: bool) -> LabelCorrectionRecord:
        nested_properties = _nested_properties(node.properties)
        contexto = str(nested_properties.get("contexto", "") or "")
        categoria = str(nested_properties.get("categoria", "") or "")
        current_label = normalize_graph_category(node.label) or "ENTIDADE"

        try:
            classification = self._classifier.classify(
                name=node.name,
                contexto=contexto,
                current_label=current_label,
            )
            predicted_label = classification.label
            justificativa = classification.justificativa
            should_update = predicted_label != current_label

            if should_update and not dry_run:
                updated_properties = dict(node.properties)
                self._graph_repo.update_entity(
                    node_id=node.node_id,
                    name=node.name,
                    label=predicted_label,
                    properties=updated_properties,
                )

            return LabelCorrectionRecord(
                node_id=node.node_id,
                name=node.name,
                current_label=current_label,
                predicted_label=predicted_label,
                categoria=categoria,
                contexto=contexto,
                would_update=should_update,
                updated=should_update and not dry_run,
                justificativa=justificativa,
            )
        except Exception as exc:
            return LabelCorrectionRecord(
                node_id=node.node_id,
                name=node.name,
                current_label=current_label,
                predicted_label=current_label,
                categoria=categoria,
                contexto=contexto,
                would_update=False,
                updated=False,
                justificativa="",
                error=str(exc),
            )


def _nested_properties(properties: dict[str, Any]) -> dict[str, Any]:
    nested = properties.get("properties")
    if isinstance(nested, dict):
        return nested
    return properties
