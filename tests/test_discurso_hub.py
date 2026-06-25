from __future__ import annotations

import unittest

from src.domain.discurso_context import DiscursoContext
from src.domain.models import Entity
from src.infrastructure.llm.llamaindex_client import (
    _build_discurso_hub_node,
    _build_discurso_hub_relationships,
)


class TestDiscursoHub(unittest.TestCase):
    def test_build_discurso_hub_node(self) -> None:
        context = DiscursoContext(discurso_id=9999, data_ocorrencia="25/06/2025")
        hub = _build_discurso_hub_node(context)

        self.assertEqual(hub.name, "discurso_id:9999")
        self.assertEqual(hub.label, "DISCURSO")
        self.assertEqual(hub.properties["categoria"], "DISCURSO")
        self.assertEqual(hub.properties["data_ocorrencia"], "25/06/2025")

    def test_build_discurso_hub_relationships(self) -> None:
        hub_name = "discurso_id:9999"
        nodes = [
            Entity(name="MARQUITO", label="ENTIDADE", properties={"categoria": "PESSOA"}),
            Entity(name="ESPORTE", label="TEMA", properties={"categoria": "TEMA"}),
            Entity(
                name=hub_name,
                label="DISCURSO",
                properties={"categoria": "DISCURSO", "data_ocorrencia": "25/06/2025"},
            ),
        ]

        relationships = _build_discurso_hub_relationships(nodes, hub_name)

        self.assertEqual(len(relationships), 2)
        for rel in relationships:
            self.assertEqual(rel.target, hub_name)
            self.assertEqual(rel.relation, "OCORRE_EM")
            self.assertIn(rel.source, {"MARQUITO", "ESPORTE"})

    def test_build_discurso_hub_relationships_deduplicates(self) -> None:
        hub_name = "discurso_id:42"
        nodes = [
            Entity(name="TEMA_A", label="TEMA", properties={"categoria": "TEMA"}),
            Entity(name="TEMA_A", label="TEMA", properties={"categoria": "TEMA"}),
        ]

        relationships = _build_discurso_hub_relationships(nodes, hub_name)

        self.assertEqual(len(relationships), 1)
        self.assertEqual(relationships[0].source, "TEMA_A")
        self.assertEqual(relationships[0].target, hub_name)


if __name__ == "__main__":
    unittest.main()
