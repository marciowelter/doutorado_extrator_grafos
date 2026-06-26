from __future__ import annotations

import json
import re
from typing import Any

import google.generativeai as genai
from google.api_core import exceptions as google_exceptions
import httpx
from pydantic import BaseModel, Field

from config.settings import settings
from src.infrastructure.llm.llamaindex_client import (
    StructuredThemeExtraction,
    _GeminiRateLimiter,
    _estimate_token_count,
    _is_transient_llm_error,
)


MAX_ATA_TEMAS = 10

_GEMINI_THEME_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "temas": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "nome": {"type": "string"},
                    "contexto": {"type": "string"},
                },
                "required": ["nome"],
            },
        }
    },
    "required": ["temas"],
}


class AtaThemeExtractionPayload(BaseModel):
    temas_principais: list[str] = Field(default_factory=list)


def _normalize_whitespace(value: str) -> str:
    return re.sub(r"\s+", " ", (value or "")).strip()


def _deduplicate_themes(raw_themes: list[str], limit: int = MAX_ATA_TEMAS) -> list[str]:
    seen: dict[str, str] = {}
    ordered: list[str] = []
    for raw in raw_themes:
        theme = _normalize_whitespace(raw)
        if not theme:
            continue
        key = theme.casefold()
        if key in seen:
            continue
        seen[key] = theme
        ordered.append(theme)
        if len(ordered) >= limit:
            break
    return ordered


def _parse_themes_payload(payload: object) -> list[str] | None:
    if payload is None:
        return None

    if isinstance(payload, AtaThemeExtractionPayload):
        return _deduplicate_themes(list(payload.temas_principais))

    if isinstance(payload, StructuredThemeExtraction):
        return _deduplicate_themes([item.nome for item in payload.temas if item.nome.strip()])

    if isinstance(payload, str):
        text = payload.strip()
        if not text:
            return None
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"\s*```$", "", text).strip()
        try:
            return _parse_themes_payload(json.loads(text))
        except json.JSONDecodeError:
            return None

    if isinstance(payload, dict):
        if "temas_principais" in payload:
            raw = payload.get("temas_principais")
            if isinstance(raw, list):
                return _deduplicate_themes([str(item) for item in raw])
        if "temas" in payload:
            raw = payload.get("temas")
            if isinstance(raw, list):
                names: list[str] = []
                for item in raw:
                    if isinstance(item, dict):
                        names.append(str(item.get("nome") or ""))
                    else:
                        names.append(str(item))
                return _deduplicate_themes(names)

    return None


def build_ata_discurso_prompt(titulo: str, como: str, porque: str, texto: str) -> str:
    titulo_linha = _normalize_whitespace(titulo) or "Sem título"
    como_linha = _normalize_whitespace(como) or "Não informado"
    porque_linha = _normalize_whitespace(porque) or "Não informado"
    corpo = _normalize_whitespace(texto)
    return (
        "Você é um analista de dados legislativos. "
        "Extraia os principais temas tratados no discurso parlamentar abaixo. "
        "Retorne APENAS JSON válido com a chave temas_principais contendo de 3 a 10 temas "
        "curtos e objetivos, em português brasileiro. Não inclua markdown.\n\n"
        f"Título/contexto: {titulo_linha}\n"
        f"Como: {como_linha}\n"
        f"Porque: {porque_linha}\n"
        f"Texto do discurso:\n{corpo}"
    )


class AtaThemeExtractor:
    def __init__(self, gemini_model: str | None = None) -> None:
        self._gemini_model = gemini_model or settings.gemini_model
        self._gemini_client = None
        self._rate_limiter = _GeminiRateLimiter(
            max_requests_per_minute=settings.gemini_max_requests_per_minute,
            max_requests_per_day=settings.gemini_max_requests_per_day,
            max_tokens_per_minute=settings.gemini_max_tokens_per_minute,
        )

    def _ensure_gemini_client(self) -> bool:
        if self._gemini_client is not None:
            return True

        api_key = settings.gemini_api_key.strip()
        if not api_key:
            return False

        genai.configure(api_key=api_key)
        self._gemini_client = genai.GenerativeModel(self._gemini_model)
        return True

    def extract_themes(
        self,
        *,
        titulo: str,
        como: str,
        porque: str,
        texto: str,
    ) -> list[str]:
        prompt = build_ata_discurso_prompt(titulo, como, porque, texto)
        if not _normalize_whitespace(texto):
            return []

        themes = self._extract_via_gemini(prompt)
        if themes:
            return themes

        return self._extract_via_bertopic_fallback(texto)

    def _extract_via_gemini(self, prompt: str) -> list[str]:
        if not self._ensure_gemini_client() or self._gemini_client is None:
            return []

        estimated_tokens = _estimate_token_count(prompt) + settings.gemini_max_output_tokens
        self._rate_limiter.wait_for_slot(estimated_tokens)

        max_attempts = max(1, settings.ollama_retry_attempts)
        last_error: Exception | None = None

        for attempt in range(1, max_attempts + 1):
            try:
                response = self._gemini_client.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=0,
                        max_output_tokens=settings.gemini_max_output_tokens,
                        response_mime_type="application/json",
                        response_schema=AtaThemeExtractionPayload.model_json_schema(),
                    ),
                    request_options={"timeout": settings.gemini_timeout},
                )
                parsed = getattr(response, "parsed", None)
                if parsed is not None:
                    if hasattr(parsed, "model_dump"):
                        themes = _parse_themes_payload(parsed.model_dump())
                    else:
                        themes = _parse_themes_payload(parsed)
                    if themes:
                        return themes

                model_text = (getattr(response, "text", None) or "").strip()
                themes = _parse_themes_payload(model_text)
                if themes:
                    return themes

                if model_text:
                    themes = _parse_themes_payload(json.loads(model_text))
                    if themes:
                        return themes
                return []
            except (
                google_exceptions.DeadlineExceeded,
                google_exceptions.ResourceExhausted,
                google_exceptions.ServiceUnavailable,
                google_exceptions.InternalServerError,
                google_exceptions.TooManyRequests,
                httpx.TimeoutException,
                httpx.TransportError,
            ) as exc:
                last_error = exc
                if attempt >= max_attempts:
                    break
            except Exception as exc:
                if _is_transient_llm_error(exc):
                    last_error = exc
                    if attempt >= max_attempts:
                        break
                    continue
                return []

        if last_error is not None:
            return []
        return []

    def _extract_via_bertopic_fallback(self, texto: str) -> list[str]:
        from src.infrastructure.llm.llamaindex_client import LlamaIndexKnowledgeExtractor

        extractor = LlamaIndexKnowledgeExtractor()
        structured, _ = extractor._extract_themes_via_bertopic(texto)
        if structured is None:
            return []
        return _deduplicate_themes([item.nome for item in structured.temas if item.nome.strip()])
