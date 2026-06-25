from __future__ import annotations

from contextlib import contextmanager, redirect_stderr, redirect_stdout
from datetime import datetime, timedelta, timezone
import io
import json
import os
import re
from collections import deque
import threading
import time
from typing import Any, Iterator

from bertopic import BERTopic
from google.api_core import exceptions as google_exceptions
import google.generativeai as genai
from gliner import GLiNER
import httpx
from llama_index.core import Settings
from llama_index.core.prompts import PromptTemplate
from llama_index.llms.ollama import Ollama
from pydantic import BaseModel, Field
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from config.settings import settings
from src.domain.discurso_context import DiscursoContext
from src.domain.models import Entity, KnowledgeGraphExtraction, Relationship
from src.domain.normalization import (
    normalize_for_match,
    normalize_graph_category,
    normalize_graph_name,
)
from src.domain.repositories import KnowledgeExtractor


MAX_DEBUG_CHARS = 400
GLINER_MAX_WORDS = 380  # margem abaixo do limite de 384 tokens do GLiNER
LLM_NOT_CONFIGURED_ERROR = "Settings.llm is not configured"
_LAST_EXTRACTION_DEBUG: dict[str, Any] = {"attempts": [], "fallback_used": False}


THEME_EXTRACTION_PROMPT = PromptTemplate(
    """
Voce e um especialista em extracao de temas e assuntos em textos de portugues brasileiro.
Extraia uma lista de temas tratados no texto e devolva SOMENTE JSON valido seguindo o schema.

Regras obrigatorias:
1) Nao invente temas; use apenas o que estiver no texto.
2) O nome do tema deve ser curto e claro (uma frase curta ou sintagma nominal), por exemplo: EDUCAÇÃO, SAÚDE, AGRICULTURA, SEGURANÇA PÚBLICA, MEIO AMBIENTE, CULTURA.
3) contexto deve ser curto (uma frase) explicando por que o tema e relevante no texto.
4) Se nao houver temas identificaveis, retorne lista vazia.

Texto:
{text}
""".strip()
)


class StructuredTheme(BaseModel):
    nome: str = Field(min_length=1, description="Nome curto do tema")
    contexto: str | None = Field(default=None, description="Resumo curto da relevancia (opcional)")


class StructuredThemeExtraction(BaseModel):
    temas: list[StructuredTheme] = Field(default_factory=list)


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


_GEMINI_JSON_REPAIR_PROMPT = (
    "Converta o conteudo abaixo para JSON VALIDO no schema: "
    '{"temas":[{"nome":"string","contexto":"string opcional"}]}. '
    "Retorne SOMENTE JSON valido, sem markdown e sem explicacoes.\n\n"
    "CONTEUDO:\n{content}"
)


_THEME_HINTS = {"tema", "assunto", "topico", "subject", "subjecto"}

_BERTOPIC_STOPWORDS = {
    "de",
    "da",
    "do",
    "das",
    "dos",
    "a",
    "o",
    "as",
    "os",
    "e",
    "em",
    "na",
    "no",
    "nas",
    "nos",
    "que",
    "para",
    "por",
    "com",
    "uma",
    "um",
    "é",
    "ao",
    "aos",
    "à",
    "às",
    "se",
    "não",
    "mais",
    "como",
    "quando",
    "onde",
    "porque",
    "quem",
    "oque",
    "também",
    "foi",
    "ela",
    "aqui",
    "sobre",
    "então",
    "tem",
}


def _configure_hf_runtime() -> None:
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    try:
        from huggingface_hub.utils import disable_progress_bars

        disable_progress_bars()
    except Exception:
        pass

    try:
        from transformers.utils import logging as transformers_logging

        transformers_logging.set_verbosity_error()
        transformers_logging.disable_progress_bar()
    except Exception:
        pass


@contextmanager
def _suppress_external_model_output() -> Iterator[None]:
    sink = io.StringIO()
    with redirect_stdout(sink), redirect_stderr(sink):
        yield


_configure_hf_runtime()


def _preview(value: str, max_chars: int = MAX_DEBUG_CHARS) -> str:
    sanitized = value.replace("\n", "\\n")
    if len(sanitized) <= max_chars:
        return sanitized
    return sanitized[:max_chars] + "..."


def _set_last_extraction_debug(payload: dict[str, Any]) -> None:
    global _LAST_EXTRACTION_DEBUG
    _LAST_EXTRACTION_DEBUG = payload


def get_last_extraction_debug() -> dict[str, Any]:
    return _LAST_EXTRACTION_DEBUG


def _normalize_for_match(value: str) -> str:
    return normalize_for_match(value)


def _source_contains_candidate(source_text: str, candidate: str) -> bool:
    normalized_source = _normalize_for_match(source_text)
    normalized_candidate = _normalize_for_match(candidate)
    if len(normalized_candidate) < 2:
        return False
    return normalized_candidate in normalized_source


def _label_from_category(category: str) -> str:
    normalized = _normalize_for_match(category)
    if any(hint in normalized for hint in _THEME_HINTS):
        return "TEMA"
    return "ENTIDADE"


def _merge_nodes(themes: list[Entity], entities: list[Entity]) -> list[Entity]:
    merged_by_name: dict[str, Entity] = {}

    # Prioriza temas quando o mesmo nome aparecer em ambos.
    for node in [*themes, *entities]:
        normalized = _normalize_for_match(node.name)
        if not normalized:
            continue
        if normalized in merged_by_name:
            continue
        merged_by_name[normalized] = node

    return list(merged_by_name.values())


def _parse_gliner_labels(raw_value: str) -> list[str]:
    labels = [part.strip() for part in raw_value.split(",") if part.strip()]
    return labels or ["pessoa", "organizacao", "local", "data", "evento", "valor"]


def _split_text_into_chunks(text: str, max_words: int = GLINER_MAX_WORDS) -> list[tuple[str, int]]:
    """Divide *text* em fatias de no máximo *max_words* palavras preservando fronteiras
    de palavras.  Retorna lista de (chunk_text, char_offset) onde char_offset é a
    posição do primeiro caractere do chunk dentro de *text*."""
    words = text.split(" ")
    chunks: list[tuple[str, int]] = []
    start_word = 0
    char_offset = 0

    while start_word < len(words):
        end_word = min(start_word + max_words, len(words))
        chunk = " ".join(words[start_word:end_word])
        chunks.append((chunk, char_offset))
        char_offset += len(chunk) + 1  # +1 pelo espaço entre chunks
        start_word = end_word

    return chunks


def _extract_entity_context(text: str, start: int | None, end: int | None, radius: int = 60) -> str:
    if start is None or end is None:
        return ""
    left = max(0, start - radius)
    right = min(len(text), end + radius)
    snippet = text[left:right].strip()
    snippet = " ".join(snippet.split())
    return snippet


def _split_text_for_bertopic(text: str) -> list[str]:
    compact = " ".join(text.split())
    if not compact:
        return []

    sentence_like = [
        part.strip()
        for part in re.split(r"(?<=[\.\!\?\;\:])\s+", compact)
        if part.strip()
    ]
    sentence_like = [part for part in sentence_like if len(part) >= 50]
    if len(sentence_like) >= 2:
        return sentence_like

    chunks = _split_text_into_chunks(compact, max_words=70)
    return [chunk_text.strip() for chunk_text, _ in chunks if chunk_text.strip()]


def _fallback_theme_terms_from_text(text: str, stopwords: set[str], limit: int = 6) -> list[str]:
    compact = " ".join(text.split())
    if not compact:
        return []

    vectorizer = TfidfVectorizer(
        stop_words=sorted(stopwords),
        ngram_range=(1, 2),
        min_df=1,
    )
    matrix = vectorizer.fit_transform([compact])
    if matrix.shape[1] == 0:
        return []

    feature_names = vectorizer.get_feature_names_out()
    row = matrix[0]
    term_scores = list(zip(row.indices, row.data))
    term_scores.sort(key=lambda item: item[1], reverse=True)
    return [feature_names[term_idx] for term_idx, _ in term_scores[:limit]]


def _build_fixed_relationships(entities: list[Entity], themes: list[Entity]) -> list[Relationship]:
    relationships: list[Relationship] = []
    seen: set[tuple[str, str]] = set()

    for entity in entities:
        for theme in themes:
            if entity.name == theme.name:
                continue
            key = (entity.name, theme.name)
            if key in seen:
                continue
            seen.add(key)
            relationships.append(
                Relationship(
                    source=entity.name,
                    target=theme.name,
                    relation="RELACIONA",
                    properties={},
                )
            )

    return relationships


def _build_discurso_hub_node(discurso_context: DiscursoContext) -> Entity:
    return Entity(
        name=discurso_context.hub_name,
        label="DISCURSO",
        properties={
            "categoria": "DISCURSO",
            "data_ocorrencia": discurso_context.data_ocorrencia,
        },
    )


def _build_discurso_hub_relationships(
    nodes: list[Entity],
    hub_name: str,
) -> list[Relationship]:
    relationships: list[Relationship] = []
    seen: set[tuple[str, str]] = set()

    for node in nodes:
        if node.name == hub_name:
            continue
        key = (node.name, hub_name)
        if key in seen:
            continue
        seen.add(key)
        relationships.append(
            Relationship(
                source=node.name,
                target=hub_name,
                relation="OCORRE_EM",
                properties={},
            )
        )

    return relationships


def _build_theme_entity(raw_name: str, contexto: str) -> Entity | None:
    name = normalize_graph_name(raw_name.strip())
    if not name:
        return None

    return Entity(
        name=name,
        label="TEMA",
        properties={
            "categoria": "TEMA",
            "contexto": contexto,
        },
    )


def _is_transient_llm_error(error: Exception) -> bool:
    if isinstance(
        error,
        (
            google_exceptions.DeadlineExceeded,
            google_exceptions.ResourceExhausted,
            google_exceptions.ServiceUnavailable,
            google_exceptions.InternalServerError,
            google_exceptions.TooManyRequests,
        ),
    ):
        return True

    if isinstance(
        error,
        (
            httpx.TimeoutException,
            httpx.TransportError,
            TimeoutError,
            ConnectionError,
        ),
    ):
        return True

    lowered = str(error).lower()
    transient_hints = (
        "timed out",
        "timeout",
        "unexpected eof",
        "connection reset",
        "temporarily unavailable",
        "connection aborted",
    )
    return any(hint in lowered for hint in transient_hints)


def _estimate_token_count(text: str) -> int:
    # Heurística conservadora para PT-BR: ~4 caracteres por token.
    return max(1, len(text) // 4)


class _GeminiRateLimiter:
    def __init__(
        self,
        max_requests_per_minute: int,
        max_requests_per_day: int,
        max_tokens_per_minute: int,
    ) -> None:
        self._max_requests_per_minute = max(1, int(max_requests_per_minute))
        self._max_requests_per_day = max(1, int(max_requests_per_day))
        self._max_tokens_per_minute = max(1, int(max_tokens_per_minute))
        self._current_minute_bucket: int | None = None
        self._requests_in_current_minute = 0
        self._current_day_bucket: str | None = None
        self._requests_in_current_day = 0
        self._token_entries: deque[tuple[float, int]] = deque()
        self._lock = threading.Lock()

    def wait_for_slot(self, estimated_tokens: int) -> None:
        requested_tokens = max(1, int(estimated_tokens))
        if requested_tokens > self._max_tokens_per_minute:
            requested_tokens = self._max_tokens_per_minute

        while True:
            sleep_seconds = 0.0

            with self._lock:
                now = time.time()
                self._refresh_request_windows(now)
                self._evict_old_token_entries(now)

                tokens_in_window = sum(tokens for _, tokens in self._token_entries)

                minute_wait = 0.0
                day_wait = 0.0
                token_wait = 0.0

                if self._requests_in_current_minute >= self._max_requests_per_minute:
                    minute_wait = self._seconds_until_next_minute(now)

                if self._requests_in_current_day >= self._max_requests_per_day:
                    day_wait = self._seconds_until_next_utc_day(now)

                if tokens_in_window + requested_tokens > self._max_tokens_per_minute and self._token_entries:
                    oldest_token_ts = self._token_entries[0][0]
                    token_wait = max(0.0, 60.0 - (now - oldest_token_ts))

                sleep_seconds = max(minute_wait, day_wait, token_wait)
                if sleep_seconds <= 0.0:
                    self._requests_in_current_minute += 1
                    self._requests_in_current_day += 1
                    self._token_entries.append((now, requested_tokens))
                    return

            if sleep_seconds > 0.0:
                time.sleep(sleep_seconds)

    def _refresh_request_windows(self, now: float) -> None:
        current_minute_bucket = int(now // 60)
        if self._current_minute_bucket != current_minute_bucket:
            self._current_minute_bucket = current_minute_bucket
            self._requests_in_current_minute = 0

        current_day_bucket = datetime.fromtimestamp(now, tz=timezone.utc).date().isoformat()
        if self._current_day_bucket != current_day_bucket:
            self._current_day_bucket = current_day_bucket
            self._requests_in_current_day = 0

    def _evict_old_token_entries(self, now: float) -> None:
        while self._token_entries and now - self._token_entries[0][0] >= 60.0:
            self._token_entries.popleft()

    def _seconds_until_next_minute(self, now: float) -> float:
        return max(0.0, (int(now // 60) + 1) * 60 - now)

    def _seconds_until_next_utc_day(self, now: float) -> float:
        current = datetime.fromtimestamp(now, tz=timezone.utc)
        next_day = (current + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        return max(0.0, (next_day - current).total_seconds())


def configure_llamaindex(model: str | None = None) -> None:
    llm = Ollama(
        model=model or settings.ollama_model,
        base_url=settings.ollama_base_url,
        request_timeout=settings.ollama_timeout,
        keep_alive=settings.ollama_keep_alive,
        temperature=0,
    )

    Settings.llm = llm


class LlamaIndexKnowledgeExtractor(KnowledgeExtractor):
    def __init__(
        self,
        theme_provider: str | None = None,
        ollama_model: str | None = None,
        gemini_model: str | None = None,
    ) -> None:
        self._theme_provider = "bertopic"
        self._ollama_model = ollama_model or settings.ollama_model
        self._gemini_model = gemini_model or settings.gemini_model

        self._gemini_model_client = None

        self._gemini_rate_limiter = _GeminiRateLimiter(
            max_requests_per_minute=settings.gemini_max_requests_per_minute,
            max_requests_per_day=settings.gemini_max_requests_per_day,
            max_tokens_per_minute=settings.gemini_max_tokens_per_minute,
        )

        with _suppress_external_model_output():
            self._ner_model = GLiNER.from_pretrained(settings.gliner_model)
        self._gliner_labels = _parse_gliner_labels(settings.gliner_labels)
        self._bertopic_vectorizer = CountVectorizer(
            stop_words=sorted(_BERTOPIC_STOPWORDS),
            ngram_range=(1, 2),
            min_df=1,
        )
        self._bertopic_model: BERTopic | None = None
        self._bertopic_lock = threading.Lock()

    def _get_bertopic_model(self) -> BERTopic:
        if self._bertopic_model is not None:
            return self._bertopic_model

        with self._bertopic_lock:
            if self._bertopic_model is None:
                with _suppress_external_model_output():
                    self._bertopic_model = BERTopic(
                        language="multilingual",
                        min_topic_size=2,
                        calculate_probabilities=False,
                        verbose=False,
                        vectorizer_model=self._bertopic_vectorizer,
                    )

        return self._bertopic_model

    def _provider_model_name(self) -> str:
        return "BERTopic"

    def _extract_themes_via_bertopic(
        self,
        text: str,
    ) -> tuple[StructuredThemeExtraction | None, Exception | None]:
        docs = _split_text_for_bertopic(text)
        if len(docs) < 2:
            return StructuredThemeExtraction(temas=[]), None

        try:
            topic_model = self._get_bertopic_model()
            with self._bertopic_lock:
                topics, _ = topic_model.fit_transform(docs)
                topic_info = topic_model.get_topic_info()

                themes: list[StructuredTheme] = []
                seen_names: set[str] = set()

                for _, row in topic_info.iterrows():
                    topic_id = int(row["Topic"])
                    if topic_id == -1:
                        continue

                    topic_terms = [word for word, _score in (topic_model.get_topic(topic_id) or [])[:4]]
                    if not topic_terms:
                        continue

                    theme_name = normalize_graph_name(topic_terms[0])
                    if not theme_name or theme_name in seen_names:
                        continue

                    context = ""
                    for doc, assigned_topic in zip(docs, topics):
                        if int(assigned_topic) == topic_id:
                            context = doc[:220]
                            break

                    themes.append(StructuredTheme(nome=theme_name, contexto=context))
                    seen_names.add(theme_name)

            if not themes:
                fallback_terms = _fallback_theme_terms_from_text(text, _BERTOPIC_STOPWORDS)
                themes = [StructuredTheme(nome=term, contexto="") for term in fallback_terms]

            return StructuredThemeExtraction(temas=themes), None
        except Exception as exc:
            fallback_terms = _fallback_theme_terms_from_text(text, _BERTOPIC_STOPWORDS)
            if fallback_terms:
                return StructuredThemeExtraction(temas=[StructuredTheme(nome=term, contexto="") for term in fallback_terms]), exc
            return None, exc

    def _extract_themes_via_ollama(
        self,
        text: str,
    ) -> tuple[StructuredThemeExtraction | None, Exception | None]:
        if Settings.llm is None:
            raise RuntimeError(LLM_NOT_CONFIGURED_ERROR)

        structured: StructuredThemeExtraction | None = None
        last_error: Exception | None = None
        max_attempts = max(1, settings.ollama_retry_attempts)

        for attempt in range(1, max_attempts + 1):
            try:
                structured = Settings.llm.structured_predict(
                    StructuredThemeExtraction,
                    THEME_EXTRACTION_PROMPT,
                    text=text,
                )
                break
            except Exception as exc:
                last_error = exc
                if not _is_transient_llm_error(exc):
                    break
                if attempt >= max_attempts:
                    break
                delay_seconds = max(0.0, settings.ollama_retry_delay_seconds) * attempt
                if delay_seconds > 0:
                    time.sleep(delay_seconds)

        return structured, last_error

    def _extract_themes_via_gemini(
        self,
        text: str,
    ) -> tuple[StructuredThemeExtraction | None, Exception | None]:
        api_key = settings.gemini_api_key.strip()
        if not api_key:
            return None, RuntimeError("GEMINI_API_KEY nao configurada")
        if self._gemini_model_client is None:
            return None, RuntimeError("Modelo Gemini nao inicializado")

        prompt_text = THEME_EXTRACTION_PROMPT.format(text=text)
        estimated_tokens = _estimate_token_count(prompt_text) + settings.gemini_max_output_tokens
        self._gemini_rate_limiter.wait_for_slot(estimated_tokens)

        last_error: Exception | None = None
        max_attempts = max(1, settings.ollama_retry_attempts)

        for attempt in range(1, max_attempts + 1):
            try:
                response = self._gemini_model_client.generate_content(
                    prompt_text,
                    generation_config=genai.types.GenerationConfig(
                        temperature=0,
                        max_output_tokens=settings.gemini_max_output_tokens,
                        response_mime_type="application/json",
                        response_schema=_GEMINI_THEME_RESPONSE_SCHEMA,
                    ),
                    request_options={"timeout": settings.gemini_timeout},
                )
                return self._parse_gemini_structured_response(response), None
            except Exception as exc:
                last_error = exc
                if not _is_transient_llm_error(exc):
                    break
                if attempt >= max_attempts:
                    break
                time.sleep(max(0.0, settings.ollama_retry_delay_seconds) * attempt)

        return None, last_error

    def _parse_gemini_structured_response(self, response: Any) -> StructuredThemeExtraction:
        model_text = self._extract_gemini_text_response(response)
        if not model_text:
            return StructuredThemeExtraction(temas=[])

        try:
            parsed_payload = self._normalize_partial_theme_payload(self._parse_json_like_payload(model_text))
        except Exception as first_error:
            repaired_text = self._repair_theme_json_with_gemini(model_text)
            try:
                parsed_payload = self._normalize_partial_theme_payload(self._parse_json_like_payload(repaired_text))
            except Exception as repair_error:
                raise ValueError(
                    (
                        "Falha ao interpretar JSON de temas no Gemini "
                        f"(erro inicial: {first_error}; erro reparo: {repair_error})"
                    )
                ) from repair_error

        return StructuredThemeExtraction.model_validate(parsed_payload)

    def _repair_theme_json_with_gemini(self, raw_content: str) -> str:
        if self._gemini_model_client is None:
            raise RuntimeError("Modelo Gemini nao inicializado para reparo de JSON")

        repair_prompt = _GEMINI_JSON_REPAIR_PROMPT.format(content=raw_content)
        estimated_tokens = _estimate_token_count(repair_prompt) + settings.gemini_max_output_tokens
        self._gemini_rate_limiter.wait_for_slot(estimated_tokens)

        repair_response = self._gemini_model_client.generate_content(
            repair_prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0,
                max_output_tokens=settings.gemini_max_output_tokens,
                response_mime_type="application/json",
                response_schema=_GEMINI_THEME_RESPONSE_SCHEMA,
            ),
            request_options={"timeout": settings.gemini_timeout},
        )
        return self._extract_gemini_text_response(repair_response)

    def _extract_gemini_text_response(self, response: Any) -> str:
        text_value = getattr(response, "text", None)
        if isinstance(text_value, str) and text_value.strip():
            return text_value.strip()

        candidates = getattr(response, "candidates", None) or []
        if not candidates:
            return ""

        first_candidate = candidates[0]
        content = getattr(first_candidate, "content", None)
        parts = getattr(content, "parts", None) or []
        if not parts:
            return ""

        part_text = getattr(parts[0], "text", None)
        if isinstance(part_text, str):
            return part_text.strip()

        return ""

    def _parse_json_like_payload(self, raw_text: str) -> Any:
        sanitized = raw_text.strip()
        fenced_match = re.search(r"```(?:json)?\s*(.*?)\s*```", sanitized, flags=re.DOTALL)
        if fenced_match:
            sanitized = fenced_match.group(1).strip()

        try:
            return json.loads(sanitized)
        except json.JSONDecodeError:
            pass

        object_match = re.search(r"\{.*\}", sanitized, flags=re.DOTALL)
        if object_match:
            try:
                return json.loads(object_match.group(0))
            except json.JSONDecodeError:
                pass

        array_match = re.search(r"\[.*\]", sanitized, flags=re.DOTALL)
        if array_match:
            try:
                return json.loads(array_match.group(0))
            except json.JSONDecodeError:
                pass

        raise ValueError("Resposta do modelo nao contem JSON valido")

    def _normalize_partial_theme_payload(self, payload: Any) -> dict[str, Any]:
        if isinstance(payload, dict):
            if isinstance(payload.get("temas"), list):
                return {"temas": self._normalize_theme_items(payload.get("temas") or [])}

            candidate_lists = [
                payload.get("themes"),
                payload.get("assuntos"),
                payload.get("topicos"),
                payload.get("topics"),
            ]
            for candidate in candidate_lists:
                if isinstance(candidate, list):
                    return {"temas": self._normalize_theme_items(candidate)}

            if any(key in payload for key in ("nome", "name", "tema", "assunto", "topico", "topic")):
                return {"temas": self._normalize_theme_items([payload])}

            return {"temas": []}

        if isinstance(payload, list):
            return {"temas": self._normalize_theme_items(payload)}

        return {"temas": []}

    def _normalize_theme_items(self, items: list[Any]) -> list[dict[str, str]]:
        normalized: list[dict[str, str]] = []
        for item in items:
            if isinstance(item, str):
                name = item.strip()
                if name:
                    normalized.append({"nome": name})
                continue

            if not isinstance(item, dict):
                continue

            name = self._extract_theme_name(item)
            if not name:
                continue

            normalized_item: dict[str, str] = {"nome": name}
            context = self._extract_theme_context(item)
            if context:
                normalized_item["contexto"] = context

            normalized.append(normalized_item)

        return normalized

    def _extract_theme_name(self, item: dict[str, Any]) -> str:
        raw_name = (
            item.get("nome")
            or item.get("name")
            or item.get("tema")
            or item.get("assunto")
            or item.get("topico")
            or item.get("topic")
        )
        if raw_name is None:
            return ""
        return str(raw_name).strip()

    def _extract_theme_context(self, item: dict[str, Any]) -> str:
        raw_context = (
            item.get("contexto")
            or item.get("context")
            or item.get("descricao")
            or item.get("description")
        )
        if raw_context is None:
            return ""
        return str(raw_context).strip()

    def _parse_json_payload(self, raw_text: str) -> dict[str, Any]:
        sanitized = raw_text.strip()
        fenced_match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", sanitized, flags=re.DOTALL)
        if fenced_match:
            sanitized = fenced_match.group(1).strip()

        try:
            payload = json.loads(sanitized)
            if isinstance(payload, dict):
                return payload
        except json.JSONDecodeError:
            pass

        object_match = re.search(r"\{.*\}", sanitized, flags=re.DOTALL)
        if object_match:
            payload = json.loads(object_match.group(0))
            if isinstance(payload, dict):
                return payload

        raise ValueError("Resposta do modelo nao contem JSON de objeto valido")

    def _build_themes_by_name(
        self,
        structured: StructuredThemeExtraction | None,
        additional_themes: list[str] | None,
    ) -> dict[str, Entity]:
        themes_by_name: dict[str, Entity] = {}

        if structured is not None:
            for item in structured.temas:
                self._add_theme_if_valid(
                    target=themes_by_name,
                    raw_theme=item.nome,
                    context=(item.contexto or "").strip(),
                    overwrite=True,
                )

        if additional_themes:
            for raw_theme in additional_themes:
                self._add_theme_if_valid(
                    target=themes_by_name,
                    raw_theme=str(raw_theme),
                    context="Tema adicionado a partir da tabela datamart_oque.",
                    overwrite=False,
                )

        return themes_by_name

    def _add_theme_if_valid(
        self,
        target: dict[str, Entity],
        raw_theme: str,
        context: str,
        overwrite: bool,
    ) -> None:
        theme_entity = _build_theme_entity(raw_theme, context)
        if theme_entity is None:
            return
        if not overwrite and theme_entity.name in target:
            return
        target[theme_entity.name] = theme_entity

    def _build_theme_preview(
        self,
        provider: str,
        structured: StructuredThemeExtraction | None,
        last_error: Exception | None,
    ) -> str:
        if structured is not None:
            return _preview(f"provider={provider}; result={structured.model_dump_json(ensure_ascii=False)}")
        if last_error is not None:
            return _preview(f"provider={provider}; fallback_due_to_theme_extraction_error: {last_error}")
        return f"provider={provider}; fallback_without_theme_response"

    def _extract_themes(
        self,
        text: str,
        additional_themes: list[str] | None = None,
    ) -> tuple[list[Entity], str]:
        provider = self._theme_provider
        structured, last_error = self._extract_themes_via_bertopic(text)

        themes_by_name = self._build_themes_by_name(structured, additional_themes)
        preview = self._build_theme_preview(provider, structured, last_error)

        return list(themes_by_name.values()), preview

    def _extract_entities(self, text: str) -> tuple[list[Entity], str]:
        chunks = _split_text_into_chunks(text)

        entities_by_name: dict[str, Entity] = {}
        debug_items: list[dict[str, Any]] = []

        all_results: list[Any] = []
        for chunk_text, chunk_offset in chunks:
            chunk_result = self._ner_model.predict_entities(
                chunk_text,
                labels=self._gliner_labels,
                threshold=settings.gliner_threshold,
            )
            for raw in chunk_result:
                raw_start = raw.get("start")
                raw_end = raw.get("end")
                if isinstance(raw_start, int | float):
                    raw["start"] = int(raw_start) + chunk_offset
                if isinstance(raw_end, int | float):
                    raw["end"] = int(raw_end) + chunk_offset
                all_results.append(raw)

        for raw in all_results:
            raw_name = str(raw.get("text", "")).strip()
            if not raw_name:
                continue
            if not _source_contains_candidate(text, raw_name):
                continue

            raw_label = str(raw.get("label", "ENTIDADE")).strip()
            category = normalize_graph_category(raw_label)
            if _label_from_category(category) == "TEMA":
                # Temas continuam em etapa dedicada via BERTopic.
                continue

            name = normalize_graph_name(raw_name)
            if not name:
                continue

            score = raw.get("score")
            start = raw.get("start")
            end = raw.get("end")
            contexto = _extract_entity_context(
                text,
                int(start) if isinstance(start, int | float) else None,
                int(end) if isinstance(end, int | float) else None,
            )

            properties: dict[str, Any] = {
                "categoria": category,
                "contexto": contexto,
            }
            if isinstance(score, int | float):
                properties["score"] = round(float(score), 4)

            entities_by_name[name] = Entity(
                name=name,
                label="ENTIDADE",
                properties=properties,
            )

            debug_items.append(
                {
                    "text": raw_name,
                    "label": raw_label,
                    "score": score,
                    "start": start,
                    "end": end,
                }
            )

        preview = _preview(str(debug_items))
        return list(entities_by_name.values()), preview

    def extract(
        self,
        text: str,
        additional_themes: list[str] | None = None,
        discurso_context: DiscursoContext | None = None,
        cached_themes: list[Entity] | None = None,
    ) -> KnowledgeGraphExtraction:
        started_at = time.perf_counter()
        attempts: list[dict[str, Any]] = []

        themes_started_at = time.perf_counter()
        if cached_themes is not None:
            themes = [theme.model_copy(deep=True) for theme in cached_themes]
            themes_preview = f"cached_themes={len(themes)}"
            themes_attempt = "themes_cached_by_discurso"
        else:
            themes, themes_preview = self._extract_themes(text, additional_themes=additional_themes)
            themes_attempt = f"themes_structured_predict_{self._theme_provider}"
        themes_seconds = round(time.perf_counter() - themes_started_at, 4)
        attempts.append(
            {
                "attempt": themes_attempt,
                "ok": bool(themes),
                "raw_preview": themes_preview,
                "theme_count": len(themes),
                "additional_theme_count": len(additional_themes or []),
                "duration_seconds": themes_seconds,
                "provider": self._theme_provider,
                "model": self._provider_model_name(),
                "cached": cached_themes is not None,
            }
        )

        entities_started_at = time.perf_counter()
        entities, entities_preview = self._extract_entities(text)
        entities_seconds = round(time.perf_counter() - entities_started_at, 4)
        attempts.append(
            {
                "attempt": "entities_gliner_predict",
                "ok": bool(entities),
                "raw_preview": entities_preview,
                "entity_count": len(entities),
                "duration_seconds": entities_seconds,
            }
        )

        nodes = _merge_nodes(themes, entities)
        hub_node: Entity | None = None
        if discurso_context is not None:
            hub_node = _build_discurso_hub_node(discurso_context)
            nodes = _merge_nodes(nodes, [hub_node])

        if not nodes:
            _set_last_extraction_debug(
                {
                    "pipeline": "gliner_entities_bertopic_themes_fixed_relationships",
                    "attempts": attempts,
                    "timings": {
                        "themes_seconds": themes_seconds,
                        "entities_gliner_seconds": entities_seconds,
                        "relationships_build_seconds": 0.0,
                        "extraction_total_seconds": round(time.perf_counter() - started_at, 4),
                    },
                    "fallback_used": True,
                }
            )
            return KnowledgeGraphExtraction(entities=[], relationships=[])

        relationships_started_at = time.perf_counter()
        if discurso_context is not None and hub_node is not None:
            relationships = _build_discurso_hub_relationships(nodes, hub_node.name)
            relationship_attempt = "relationships_discurso_hub"
            relationship_preview = (
                f"hub={hub_node.name} nodes={len(nodes) - 1} "
                f"data_ocorrencia={discurso_context.data_ocorrencia}"
            )
            pipeline_name = "gliner_entities_bertopic_themes_discurso_hub"
        else:
            relationships = _build_fixed_relationships(entities=entities, themes=themes)
            relationship_attempt = "relationships_cartesian_entities_themes"
            relationship_preview = f"entities={len(entities)} themes={len(themes)}"
            pipeline_name = "gliner_entities_bertopic_themes_fixed_relationships"
        relationships_seconds = round(time.perf_counter() - relationships_started_at, 4)
        attempts.append(
            {
                "attempt": relationship_attempt,
                "ok": True,
                "raw_preview": relationship_preview,
                "triplet_count": len(relationships),
                "duration_seconds": relationships_seconds,
            }
        )

        _set_last_extraction_debug(
            {
                "pipeline": pipeline_name,
                "attempts": attempts,
                "timings": {
                    "themes_seconds": themes_seconds,
                    "entities_gliner_seconds": entities_seconds,
                    "relationships_build_seconds": relationships_seconds,
                    "extraction_total_seconds": round(time.perf_counter() - started_at, 4),
                },
                "fallback_used": False,
            }
        )

        return KnowledgeGraphExtraction(entities=nodes, relationships=relationships)
