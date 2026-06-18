from __future__ import annotations

import json
import re
import time
from typing import Literal, Protocol

import google.generativeai as genai
from llama_index.core import Settings
from llama_index.core.prompts import PromptTemplate
from llama_index.llms.ollama import Ollama
from pydantic import BaseModel, Field

from config.settings import settings
from src.domain.normalization import normalize_graph_category

LLM_NOT_CONFIGURED_ERROR = "Settings.llm is not configured"
MAX_CONTEXTO_CHARS = 400
MAX_JUSTIFICATIVA_CHARS = 240

_GEMINI_LABEL_RESPONSE_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "label": {"type": "string", "enum": ["TEMA", "ENTIDADE"]},
        "justificativa": {"type": "string"},
    },
    "required": ["label", "justificativa"],
}

_GEMINI_LABEL_JSON_REPAIR_PROMPT = (
    "Converta o conteudo abaixo para JSON VALIDO no schema: "
    '{"label":"TEMA|ENTIDADE","justificativa":"texto curto"}. '
    "Retorne SOMENTE JSON valido, sem markdown e sem explicacoes. "
    f"A justificativa deve ter no maximo {MAX_JUSTIFICATIVA_CHARS} caracteres.\n\n"
    "CONTEUDO:\n{content}"
)


LABEL_CLASSIFICATION_PROMPT = PromptTemplate(
    """
Voce e um especialista em classificacao de nos de grafos de conhecimento em portugues brasileiro.
Dado o nome e o contexto de um no, determine se ele representa um TEMA ou ENTIDADE.

TEMA: assunto, topico, area tematica ou conceito abstrato discutido no texto.
Exemplos de TEMA: SAUDE, EDUCACAO, MEIO AMBIENTE, SEGURANCA PUBLICA, MOBILIZACAO,
LUTA, CONSENSO, AGRICULTURA, GESTAO, CAMARA (como instituicao/pauta abstrata).

ENTIDADE: elemento concreto e identificavel nominalmente no texto.
Exemplos de ENTIDADE: JOAO SILVA, VINICIUS, ASSEMBLEIA LEGISLATIVA, FLORIANOPOLIS,
CODIGO DE TRANSITO, PREFEITO, VEReador RENATO, SINDICATO DOS TRABALHADORES RURAIS.

Regras obrigatorias:
1) Areas de politica publica e assuntos genericos (SAUDE, EDUCACAO, MEIO AMBIENTE, SEGURANCA)
   sao TEMA, nunca ENTIDADE.
2) Nomes proprios de pessoas (inclusive apenas o primeiro nome) sao ENTIDADE.
3) Organizacoes, locais, instituicoes, documentos legais e cargos ocupados por pessoas
   especificas sao ENTIDADE.
4) Palavras que expressam acoes, movimentos ou conceitos abstratos (MOBILIZACAO, LUTA,
   CONSENSO, GESTAO) sao TEMA, salvo se claramente forem nome de pessoa ou organizacao.
5) Use o contexto para desambiguar; em caso de duvida entre area tematica e entidade concreta,
   prefira TEMA.
6) A justificativa deve ser curta (maximo {max_justificativa} caracteres).
7) Responda SOMENTE com JSON valido seguindo o schema.

Nome: {name}
Contexto: {contexto}
Label atual no grafo: {current_label}
""".strip()
)


class LabelClassification(BaseModel):
    label: Literal["TEMA", "ENTIDADE"] = Field(description="Classificacao correta do no")
    justificativa: str = Field(min_length=1, description="Breve explicacao da classificacao")


class LabelClassifier(Protocol):
    def classify(
        self,
        *,
        name: str,
        contexto: str,
        current_label: str,
    ) -> LabelClassification:
        ...


def configure_llamaindex(model: str | None = None) -> Ollama:
    llm = Ollama(
        model=model or settings.ollama_model,
        base_url=settings.ollama_base_url,
        request_timeout=settings.ollama_timeout,
        keep_alive=settings.ollama_keep_alive,
        temperature=0,
    )
    Settings.llm = llm
    return llm


def create_label_classifier(provider: str = "ollama") -> LabelClassifier:
    normalized_provider = provider.strip().lower()
    if normalized_provider == "gemini":
        return GeminiLabelClassifier()
    if normalized_provider == "ollama":
        return OllamaLabelClassifier()
    raise ValueError(f"Provider de classificacao invalido: {provider}")


def _format_label_prompt(*, name: str, contexto: str, current_label: str) -> str:
    return LABEL_CLASSIFICATION_PROMPT.format(
        name=name,
        contexto=_truncate_text(contexto or "(sem contexto)", MAX_CONTEXTO_CHARS),
        current_label=current_label or "ENTIDADE",
        max_justificativa=MAX_JUSTIFICATIVA_CHARS,
    )


def _truncate_text(value: str, max_chars: int) -> str:
    sanitized = value.strip()
    if len(sanitized) <= max_chars:
        return sanitized
    return sanitized[: max_chars - 3].rstrip() + "..."


def _truncate_justificativa(value: str) -> str:
    sanitized = value.strip()
    if len(sanitized) <= MAX_JUSTIFICATIVA_CHARS:
        return sanitized
    return sanitized[: MAX_JUSTIFICATIVA_CHARS - 3].rstrip() + "..."


def _normalize_label_classification(raw: LabelClassification) -> LabelClassification:
    normalized_label = normalize_graph_category(raw.label)
    if normalized_label not in {"TEMA", "ENTIDADE"}:
        raise ValueError(f"Label invalida retornada pelo modelo: {raw.label}")
    justificativa = _truncate_justificativa(raw.justificativa)
    if not justificativa:
        raise ValueError("Justificativa vazia retornada pelo modelo")
    return LabelClassification(
        label=normalized_label,  # type: ignore[arg-type]
        justificativa=justificativa,
    )


def _should_retry_gemini_error(exc: Exception) -> bool:
    if _is_transient_llm_error(exc):
        return True
    message = str(exc).lower()
    retry_tokens = (
        "json valido",
        "json",
        "resposta vazia",
        "truncad",
        "malform",
        "invalid",
        "parse",
        "schema",
        "justificativa vazia",
    )
    return any(token in message for token in retry_tokens)


def _is_transient_llm_error(exc: Exception) -> bool:
    if isinstance(exc, TimeoutError | ConnectionError):
        return True
    message = str(exc).lower()
    transient_tokens = ("timeout", "503", "429", "500", "502", "504", "resource exhausted", "unavailable")
    return any(token in message for token in transient_tokens)


def _extract_gemini_text_response(response: object) -> str:
    try:
        text_value = getattr(response, "text", None)
        if isinstance(text_value, str) and text_value.strip():
            return text_value.strip()
    except Exception:
        pass

    candidates = getattr(response, "candidates", None) or []
    if not candidates:
        return ""

    chunks: list[str] = []
    for candidate in candidates:
        content = getattr(candidate, "content", None)
        parts = getattr(content, "parts", None) or []
        for part in parts:
            part_text = getattr(part, "text", None)
            if isinstance(part_text, str) and part_text.strip():
                chunks.append(part_text.strip())

    return "\n".join(chunks).strip()


def _get_gemini_finish_reason(response: object) -> str:
    candidates = getattr(response, "candidates", None) or []
    if not candidates:
        return ""
    finish_reason = getattr(candidates[0], "finish_reason", None)
    if finish_reason is None:
        return ""
    return str(finish_reason)


def _ensure_gemini_response_has_text(response: object) -> str:
    model_text = _extract_gemini_text_response(response)
    finish_reason = _get_gemini_finish_reason(response)

    if model_text:
        sanitized = _sanitize_json_like_text(model_text)
        if sanitized.startswith("{") or '"label"' in sanitized.lower():
            return model_text
        if finish_reason in {"MAX_TOKENS", "2"}:
            raise ValueError("Resposta JSON truncada pelo Gemini")

    prompt_feedback = getattr(response, "prompt_feedback", None)
    block_reason = getattr(prompt_feedback, "block_reason", None) if prompt_feedback else None
    details = ", ".join(
        item
        for item in (
            f"finish_reason={finish_reason}" if finish_reason else "",
            f"block_reason={block_reason}" if block_reason else "",
        )
        if item
    )
    suffix = f" ({details})" if details else ""
    raise ValueError(f"Gemini retornou resposta vazia ou sem JSON{suffix}")


def _sanitize_json_like_text(raw_text: str) -> str:
    sanitized = raw_text.strip()
    fenced_match = re.search(r"```(?:json)?\s*(.*?)\s*```", sanitized, flags=re.DOTALL)
    if fenced_match:
        sanitized = fenced_match.group(1).strip()

    if sanitized.lower().startswith("here is the json requested:"):
        sanitized = sanitized.split(":", 1)[1].strip()

    json_start = sanitized.find("{")
    if json_start > 0:
        sanitized = sanitized[json_start:]

    return sanitized.strip()


def _close_truncated_json_object(raw_text: str) -> str:
    sanitized = _sanitize_json_like_text(raw_text)
    if not sanitized.startswith("{"):
        return sanitized

    if sanitized.endswith("}"):
        return sanitized

    closed = sanitized.rstrip(", \n\r\t")
    if closed.count('"') % 2 == 1:
        closed += '"'
    if not closed.endswith("}"):
        closed += "}"
    return closed


def _extract_label_fields_with_regex(raw_text: str) -> dict[str, str] | None:
    sanitized = _sanitize_json_like_text(raw_text)
    label_match = re.search(
        r'"label"\s*:\s*"(TEMA|ENTIDADE)"',
        sanitized,
        flags=re.IGNORECASE,
    ) or re.search(
        r'"label"\s*:\s*(TEMA|ENTIDADE)\b',
        sanitized,
        flags=re.IGNORECASE,
    )
    if not label_match:
        return None

    justificativa_match = re.search(
        r'"justificativa"\s*:\s*"((?:\\.|[^"\\])*)"',
        sanitized,
        flags=re.DOTALL,
    )
    justificativa = ""
    if justificativa_match:
        try:
            justificativa = json.loads(f'"{justificativa_match.group(1)}"')
        except json.JSONDecodeError:
            justificativa = justificativa_match.group(1).strip()

    if not justificativa:
        justificativa = "Classificacao inferida a partir de resposta parcial do modelo."

    return {
        "label": label_match.group(1).upper(),
        "justificativa": _truncate_justificativa(justificativa),
    }


def _parse_json_like_payload(raw_text: str) -> object:
    sanitized = _sanitize_json_like_text(raw_text)
    candidates = [sanitized, _close_truncated_json_object(sanitized)]

    for candidate in candidates:
        if not candidate:
            continue
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

        object_match = re.search(r"\{.*\}", candidate, flags=re.DOTALL)
        if object_match:
            fragment = object_match.group(0)
            for variant in (fragment, _close_truncated_json_object(fragment)):
                try:
                    return json.loads(variant)
                except json.JSONDecodeError:
                    continue

    regex_payload = _extract_label_fields_with_regex(sanitized)
    if regex_payload is not None:
        return regex_payload

    raise ValueError("Resposta do modelo nao contem JSON valido")


def _parse_label_classification_from_text(raw_text: str) -> LabelClassification:
    parsed_payload = _parse_json_like_payload(raw_text)
    if isinstance(parsed_payload, dict):
        return LabelClassification.model_validate(parsed_payload)
    raise ValueError("Payload de classificacao deve ser um objeto JSON")


class OllamaLabelClassifier:
    def __init__(self) -> None:
        self._llm = configure_llamaindex()

    def classify(
        self,
        *,
        name: str,
        contexto: str,
        current_label: str,
    ) -> LabelClassification:
        max_attempts = max(1, settings.ollama_retry_attempts)
        last_error: Exception | None = None

        for attempt in range(1, max_attempts + 1):
            try:
                structured = self._llm.structured_predict(
                    LabelClassification,
                    LABEL_CLASSIFICATION_PROMPT,
                    name=name,
                    contexto=contexto or "(sem contexto)",
                    current_label=current_label or "ENTIDADE",
                )
                return _normalize_label_classification(structured)
            except Exception as exc:
                last_error = exc
                if not _is_transient_llm_error(exc) or attempt >= max_attempts:
                    break
                delay_seconds = max(0.0, settings.ollama_retry_delay_seconds) * attempt
                time.sleep(delay_seconds)

        raise RuntimeError(f"Falha ao classificar label para '{name}': {last_error}") from last_error


class GeminiLabelClassifier:
    def __init__(self) -> None:
        api_key = settings.gemini_api_key.strip()
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY nao configurada")
        genai.configure(api_key=api_key)
        self._model = genai.GenerativeModel(settings.gemini_model)

    def classify(
        self,
        *,
        name: str,
        contexto: str,
        current_label: str,
    ) -> LabelClassification:
        prompt_text = _format_label_prompt(
            name=name,
            contexto=contexto,
            current_label=current_label,
        )
        max_attempts = max(1, settings.ollama_retry_attempts)
        last_error: Exception | None = None

        for attempt in range(1, max_attempts + 1):
            try:
                return self._classify_once(prompt_text)
            except Exception as exc:
                last_error = exc
                if not _should_retry_gemini_error(exc) or attempt >= max_attempts:
                    break
                delay_seconds = max(0.0, settings.ollama_retry_delay_seconds) * attempt
                time.sleep(delay_seconds)

        raise RuntimeError(f"Falha ao classificar label para '{name}': {last_error}") from last_error

    def _classify_once(self, prompt_text: str) -> LabelClassification:
        response = self._model.generate_content(
            prompt_text,
            generation_config=genai.types.GenerationConfig(
                temperature=0,
                max_output_tokens=settings.gemini_max_output_tokens,
                response_mime_type="application/json",
                response_schema=_GEMINI_LABEL_RESPONSE_SCHEMA,
            ),
            request_options={"timeout": settings.gemini_timeout},
        )
        model_text = _ensure_gemini_response_has_text(response)
        try:
            structured = _parse_label_classification_from_text(model_text)
            return _normalize_label_classification(structured)
        except Exception as first_error:
            repaired_text = self._repair_label_json_with_gemini(model_text)
            if repaired_text:
                try:
                    structured = _parse_label_classification_from_text(repaired_text)
                    return _normalize_label_classification(structured)
                except Exception as repair_error:
                    raise ValueError(
                        "Falha ao interpretar JSON de classificacao no Gemini "
                        f"(erro inicial: {first_error}; erro reparo: {repair_error})"
                    ) from repair_error
            raise first_error

    def _repair_label_json_with_gemini(self, raw_content: str) -> str:
        repair_prompt = _GEMINI_LABEL_JSON_REPAIR_PROMPT.format(
            content=_truncate_text(raw_content, MAX_CONTEXTO_CHARS),
        )
        repair_response = self._model.generate_content(
            repair_prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0,
                max_output_tokens=settings.gemini_max_output_tokens,
                response_mime_type="application/json",
                response_schema=_GEMINI_LABEL_RESPONSE_SCHEMA,
            ),
            request_options={"timeout": settings.gemini_timeout},
        )
        return _ensure_gemini_response_has_text(repair_response)
