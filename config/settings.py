from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()


@dataclass(frozen=True)
class Settings:
    postgres_host: str = os.getenv("POSTGRES_HOST")
    postgres_port: int = int(os.getenv("POSTGRES_PORT"))
    postgres_db: str = os.getenv("POSTGRES_DB")
    postgres_user: str = os.getenv("POSTGRES_USER", "")
    postgres_password: str = os.getenv("POSTGRES_PASSWORD", "")
    postgres_schema: str = os.getenv("POSTGRES_SCHEMA")
    postgres_connect_timeout: int = int(os.getenv("POSTGRES_CONNECT_TIMEOUT", "10"))
    postgres_keepalives_idle: int = int(os.getenv("POSTGRES_KEEPALIVES_IDLE", "30"))
    postgres_keepalives_interval: int = int(os.getenv("POSTGRES_KEEPALIVES_INTERVAL", "10"))
    postgres_keepalives_count: int = int(os.getenv("POSTGRES_KEEPALIVES_COUNT", "5"))

    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL")
    ollama_model: str = os.getenv("OLLAMA_MODEL")
    ollama_timeout: float = float(os.getenv("OLLAMA_TIMEOUT", "120"))
    ollama_keep_alive: str = os.getenv("OLLAMA_KEEP_ALIVE", "30m")
    gliner_model: str = os.getenv("GLINER_MODEL", "urchade/gliner_multi-v2.1")
    gliner_threshold: float = float(os.getenv("GLINER_THRESHOLD", "0.45"))
    gliner_labels: str = os.getenv(
        "GLINER_LABELS",
        "pessoa,organizacao,local,data,evento,valor,documento,instituicao,cargo",
    )

    graph_name: str = os.getenv("AGE_GRAPH_NAME", "doutorado_extrator_grafos_graph")


settings = Settings()


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent
