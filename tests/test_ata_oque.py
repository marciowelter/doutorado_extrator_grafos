import unittest

from src.infrastructure.llm.ata_theme_extractor import (
    _deduplicate_themes,
    _parse_themes_payload,
    build_ata_discurso_prompt,
)


class AtaThemeExtractorTests(unittest.TestCase):
    def test_deduplicate_themes_preserves_order_and_limit(self) -> None:
        themes = _deduplicate_themes(
            ["Saúde", "Educação", "saúde", "Segurança Pública", "Cultura", "Meio Ambiente"],
            limit=4,
        )
        self.assertEqual(themes, ["Saúde", "Educação", "Segurança Pública", "Cultura"])

    def test_parse_temas_principais_payload(self) -> None:
        themes = _parse_themes_payload(
            {"temas_principais": ["Agricultura", "Crédito rural", "Agricultura"]}
        )
        self.assertEqual(themes, ["Agricultura", "Crédito rural"])

    def test_build_prompt_includes_context(self) -> None:
        prompt = build_ata_discurso_prompt(
            titulo="Sessão Especial",
            como="Tribuna",
            porque="Sessão Plenária Ordinária",
            texto="Debate sobre saúde pública.",
        )
        self.assertIn("Sessão Especial", prompt)
        self.assertIn("Tribuna", prompt)
        self.assertIn("Debate sobre saúde pública.", prompt)


if __name__ == "__main__":
    unittest.main()
