from __future__ import annotations

import re
import unicodedata


def normalize_for_match(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    without_accents = "".join(char for char in normalized if not unicodedata.combining(char))
    lowered = without_accents.lower().replace("_", " ")
    lowered = re.sub(r"\s+", " ", lowered)
    return lowered.strip()


def normalize_graph_name(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    without_accents = "".join(char for char in normalized if not unicodedata.combining(char))
    cleaned = re.sub(r"[^A-Za-z0-9 ]", " ", without_accents)
    compact = re.sub(r"\s+", " ", cleaned).strip()
    return compact.upper()


def normalize_graph_category(value: str) -> str:
    return normalize_graph_name(value)


def normalize_relation_label(value: str) -> str:
    normalized_name = normalize_graph_name(value)
    if not normalized_name:
        return "OCORRE_EM"
    return normalized_name.replace(" ", "_")
