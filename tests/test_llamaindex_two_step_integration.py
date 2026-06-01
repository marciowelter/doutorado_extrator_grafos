from __future__ import annotations

import json
import unittest

from src.infrastructure.llm.llamaindex_client import LlamaIndexKnowledgeExtractor


TEST_TEXT = "Dep. Marquito relata que na presente semana aconteceu uma audiência pública na Casa, para tratar sobre o cancelamento de competições esportivas no ano. Deixa claro que compreende o cancelamento de eventos por conta das enchentes, e questiona o motivo do cancelamento de outros eventos também. Esclarece que suas críticas não são pessoais ao presidente da Fesporte, enfatizando que continuará cobrando por melhorias. Tece críticas à postura do presidente da Fesporte ao final da audiência pública que deferiu xingamentos ao deputado e a outros parlamentares. Sugere que o presidente da Fesporte peça desoneração do cargo para melhorar o desempenho da federação. Relembra que no presente ano um evento esportivo não foi entregue nenhuma premiação aos competidores vencedores. Relata que conversou com o Governador Jorginho Mello que autorizou a realização do Parajasc 2023, pois há condições totais para a execução do evento. E questiona o motivo do presidente da Fesporte ter cancelado o referido evento."


class TestLlamaIndexTwoStepIntegration(unittest.TestCase):
    def test_extract_entities_and_relationships(self) -> None:
        extractor = LlamaIndexKnowledgeExtractor()
        extraction = extractor.extract(TEST_TEXT)

        self.assertIsNotNone(extraction)
        self.assertIsInstance(extraction.entities, list)
        self.assertIsInstance(extraction.relationships, list)
        self.assertGreater(len(extraction.entities), 0, "A extração deve retornar ao menos uma entidade.")

        entity_names = {entity.name.lower() for entity in extraction.entities}
        self.assertTrue(
            any("fesporte" in name for name in entity_names),
            "Esperava encontrar ao menos uma entidade relacionada a Fesporte.",
        )
        self.assertTrue(
            any("marquito" in name for name in entity_names),
            "Esperava encontrar entidade com Marquito (ex.: Dep. Marquito).",
        )

        extracted_entities = {entity.name for entity in extraction.entities if entity.label == "ENTIDADE"}
        extracted_themes = {entity.name for entity in extraction.entities if entity.label == "TEMA"}

        # As relacoes sao deterministicas: ENTIDADE -> TEMA com tipo RELACIONA.
        for rel in extraction.relationships:
            self.assertIn(rel.source, extracted_entities)
            self.assertIn(rel.target, extracted_themes)
            self.assertEqual(rel.relation, "RELACIONA")

        if extracted_entities and extracted_themes:
            expected_relationship_count = len(extracted_entities) * len(extracted_themes)
            self.assertEqual(len(extraction.relationships), expected_relationship_count)

        print("\n=== Resultado estruturado da extração (GLiNER + Ollama Temas) ===")
        print(json.dumps(extraction.model_dump(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    unittest.main()
