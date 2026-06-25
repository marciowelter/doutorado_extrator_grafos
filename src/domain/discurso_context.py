from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DiscursoContext:
    discurso_id: int
    data_ocorrencia: str

    @property
    def hub_name(self) -> str:
        return f"discurso_id:{self.discurso_id}"
