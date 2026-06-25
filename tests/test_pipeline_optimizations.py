from __future__ import annotations

import unittest

from src.application.import_texts_use_case import extract_theme_entities
from src.domain.models import Entity, KnowledgeGraphExtraction, Relationship
from src.infrastructure.database.age_repository import (
    _prepare_entities_batch,
    _prepare_relationships_batch,
)


class TestThemeCacheHelpers(unittest.TestCase):
    def test_extract_theme_entities(self) -> None:
        extraction = KnowledgeGraphExtraction(
            entities=[
                Entity(name="SAUDE", label="TEMA", properties={"categoria": "TEMA"}),
                Entity(name="JOAO", label="ENTIDADE", properties={"categoria": "PESSOA"}),
                Entity(
                    name="discurso_id:1",
                    label="DISCURSO",
                    properties={"categoria": "DISCURSO"},
                ),
            ],
            relationships=[],
        )

        themes = extract_theme_entities(extraction)

        self.assertEqual(len(themes), 1)
        self.assertEqual(themes[0].name, "SAUDE")
        self.assertIsNot(themes[0], extraction.entities[0])


class TestAgeBatchPayload(unittest.TestCase):
    def test_prepare_entities_batch_deduplicates_by_name(self) -> None:
        entities = [
            Entity(name="Tema A", label="TEMA", properties={"categoria": "TEMA"}),
            Entity(name="tema-a", label="TEMA", properties={"categoria": "TEMA"}),
            Entity(name="João", label="ENTIDADE", properties={"categoria": "PESSOA"}),
        ]

        prepared = _prepare_entities_batch(entities)

        self.assertEqual(len(prepared), 2)
        self.assertEqual(prepared[0]["name"], "TEMA A")
        self.assertEqual(prepared[1]["name"], "JOAO")

    def test_prepare_relationships_batch_groups_by_relation(self) -> None:
        relationships = [
            Relationship(source="A", target="HUB", relation="OCORRE_EM", properties={}),
            Relationship(source="B", target="HUB", relation="OCORRE_EM", properties={}),
            Relationship(source="A", target="T", relation="RELACIONA", properties={}),
        ]

        grouped = _prepare_relationships_batch(relationships)

        self.assertEqual(len(grouped["OCORRE_EM"]), 2)
        self.assertEqual(len(grouped["RELACIONA"]), 1)


if __name__ == "__main__":
    unittest.main()
