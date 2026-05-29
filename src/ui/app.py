from __future__ import annotations

import json
import time
import traceback
from datetime import datetime, timezone
from typing import Callable

import streamlit as st

from src.application.import_texts_use_case import (
    IMPORT_SQL,
    ImportProgress,
    ImportTextsUseCase,
    RetryEvent,
)
from src.application.pipeline_use_case import PipelineUseCase
from src.application.search_use_case import SearchUseCase
from src.infrastructure.llm.llamaindex_client import get_last_extraction_debug


RECOMMENDED_MAX_CHARS = 3000
HARD_WARN_CHARS = 7000
IMPORT_JOB_NAME = "Importacao Textos"


def init_session_state() -> None:
    if "last_extraction" not in st.session_state:
        st.session_state.last_extraction = None

    if "last_search_results" not in st.session_state:
        st.session_state.last_search_results = None

    if "last_processing_info" not in st.session_state:
        st.session_state.last_processing_info = None

    if "last_tech_log" not in st.session_state:
        st.session_state.last_tech_log = None

    if "last_import_summary" not in st.session_state:
        st.session_state.last_import_summary = None


@st.cache_resource(show_spinner=False)
def get_pipeline() -> PipelineUseCase:
    pipeline = PipelineUseCase()
    pipeline.bootstrap()
    return pipeline


@st.cache_resource(show_spinner=False)
def get_search() -> SearchUseCase:
    return SearchUseCase()


def process_text_sync(text: str) -> dict:
    pipeline = get_pipeline()
    extraction = pipeline.process_text(text)
    return extraction.model_dump()


def search_sync(keyword: str) -> dict[str, list[dict[str, str]]]:
    return get_search().search(keyword)


def import_texts_sync(
    on_progress: Callable[[ImportProgress], None] | None = None,
    on_retry: Callable[[RetryEvent], None] | None = None,
) -> dict[str, int]:
    use_case = ImportTextsUseCase()
    return use_case.process_all(on_progress=on_progress, on_retry=on_retry)


def log_exec(status: str, job: str, started: float, meta: dict, error: str | None = None) -> None:
    entry = {
        "status": status,
        "job": job,
        "duration_seconds": round(time.time() - started, 2),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "meta": meta,
    }
    if error is not None:
        entry["error"] = error
    st.session_state.last_tech_log = entry


def handle_extraction_submit(text: str, char_count: int) -> None:
    started = time.time()
    status_box = st.status("Processando extracao...", expanded=True)
    try:
        status_box.write("1/3 Inicializando pipeline")
        result = process_text_sync(text)
        status_box.write("2/3 Persistindo dados no banco")
        status_box.write("3/3 Finalizando")
        status_box.update(label="Extracao concluida", state="complete")
        st.session_state.last_extraction = result
        entities = result.get("entities", [])
        relationships = result.get("relationships", [])
        st.session_state.last_processing_info = {
            "status": "ok",
            "message": (
                "Extracao concluida e dados persistidos."
                if entities or relationships
                else "Processamento concluido, mas o LLM nao retornou entidades/relacionamentos validos."
            ),
        }
        log_exec(
            "ok",
            "Extracao",
            started,
            {
                "chars": char_count,
                "llm_debug": get_last_extraction_debug(),
            },
        )
    except Exception as exc:
        status_box.update(label="Falha na extracao", state="error")
        st.session_state.last_processing_info = {
            "status": "error",
            "message": str(exc),
        }
        log_exec(
            "error",
            "Extracao",
            started,
            {
                "chars": char_count,
                "llm_debug": get_last_extraction_debug(),
            },
            str(exc),
        )
        st.error(f"Falha no processamento: {exc}")


def handle_import_submit() -> None:
    started = time.time()
    status_box = st.status("Iniciando importacao de textos...", expanded=True)
    progress_bar = st.progress(0)
    progress_text = st.empty()

    try:
        status_box.write("Conectando ao banco e carregando pendencias")

        def on_retry(event: RetryEvent) -> None:
            status_box.write(
                (
                    "Instabilidade de conexao detectada em "
                    f"{event.context}. Tentativa {event.attempt}/{event.max_attempts}; "
                    "reconectando em 2 segundos."
                )
            )

        def on_progress(event: ImportProgress) -> None:
            percentage = int((event.attempted / event.total) * 100) if event.total else 100
            progress_bar.progress(percentage)
            progress_text.caption(
                (
                    f"Processados: {event.attempted}/{event.total} | "
                    f"Sucesso: {event.successful} | Falha: {event.failed} | "
                    f"Trecho atual: {event.record.trecho_id}"
                )
            )
            status_box.write(
                (
                    f"Trecho {event.record.trecho_id} do discurso {event.record.discurso_id} "
                    "processado."
                )
            )

        summary = import_texts_sync(on_progress=on_progress, on_retry=on_retry)
        progress_bar.progress(100)
        progress_text.caption(
            (
                f"Finalizado. Processados: {summary['attempted']}/{summary['total']} | "
                f"Sucesso: {summary['successful']} | Falha: {summary['failed']}"
            )
        )
        status_box.update(label="Importacao concluida", state="complete")
        st.session_state.last_import_summary = summary
        log_exec("ok", IMPORT_JOB_NAME, started, summary)
    except Exception as exc:
        status_box.update(label="Falha na importacao", state="error")
        st.session_state.last_import_summary = {
            "total": 0,
            "attempted": 0,
            "successful": 0,
            "failed": 1,
            "error": str(exc),
        }
        log_exec("error", IMPORT_JOB_NAME, started, {}, str(exc))
        st.error(f"Falha na importacao: {exc}")


def render_experiments_tab() -> None:
    st.subheader("Extracao em tempo real")
    with st.form("exp_form", clear_on_submit=False):
        text = st.text_area("Texto para extrair", height=260, key="exp_text")
        submitted_exp = st.form_submit_button("Processar e Extrair Grafo", type="primary")

    char_count = len(text or "")
    st.caption(f"Tamanho do texto: {char_count} caracteres")
    if char_count > HARD_WARN_CHARS:
        st.warning(
            "Texto muito longo para modelo leve; pode causar demora/timeout. Recomendado processar em blocos menores."
        )
    elif char_count > RECOMMENDED_MAX_CHARS:
        st.info("Texto acima do recomendado. Considere reduzir para melhorar tempo de resposta.")

    if submitted_exp:
        if not text.strip():
            st.warning("Informe um texto antes de processar.")
        else:
            handle_extraction_submit(text, char_count)

    if st.session_state.last_processing_info is not None:
        info = st.session_state.last_processing_info
        if info["status"] == "ok":
            st.info(info["message"])
        else:
            st.warning(f"Ultimo processamento com erro: {info['message']}")

    if st.session_state.last_extraction is not None:
        st.code(
            json.dumps(st.session_state.last_extraction, ensure_ascii=False, indent=2),
            language="json",
        )


def render_search_tab() -> None:
    st.subheader("Busca no grafo")
    with st.form("search_form", clear_on_submit=False):
        keyword = st.text_input("Palavra-chave", key="search_keyword")
        submitted_search = st.form_submit_button("Buscar", type="secondary")

    if submitted_search:
        if not keyword.strip():
            st.warning("Digite uma palavra-chave.")
        else:
            started = time.time()
            status_box = st.status("Executando busca...", expanded=True)
            try:
                status_box.write("1/2 Consultando conexoes no grafo")
                results = search_sync(keyword)
                status_box.write("2/2 Filtrando relacoes de tema/assunto")
                status_box.update(label="Busca concluida", state="complete")
                st.session_state.last_search_results = results
                log_exec("ok", "Busca", started, {"keyword": keyword})
            except Exception as exc:
                status_box.update(label="Falha na busca", state="error")
                log_exec("error", "Busca", started, {"keyword": keyword}, str(exc))
                st.error(f"Falha na busca: {exc}")

    render_search_results(st.session_state.last_search_results)


def render_import_tab() -> None:
    st.subheader("Importacao de textos do Postgres")
    st.caption(
        "Carrega textos pendentes do banco 'banco' (schema 'doutorado'), processa no grafo e marca trecho.grafo=true."
    )

    with st.expander("SQL de importacao", expanded=False):
        st.code(IMPORT_SQL, language="sql")

    if st.button("Importar Textos e Processar Grafo", type="primary"):
        handle_import_submit()

    if st.session_state.last_import_summary is not None:
        st.markdown("### Ultimo resultado de importacao")
        st.json(st.session_state.last_import_summary)


def render_search_results(results: dict[str, list[dict[str, str]]] | None) -> None:
    if results is None:
        st.info("Nenhum dado carregado ainda. Informe uma palavra-chave e clique em Buscar.")
        return

    st.markdown("### Conexoes no grafo (Apache AGE)")
    if not results["graph"]:
        st.info("Nenhuma conexao encontrada.")
    else:
        st.markdown("#### Relacoes de tema/assunto")
        if not results.get("graph_theme"):
            st.info("Nenhuma relacao de tema/assunto encontrada para este termo.")
        else:
            st.dataframe(results["graph_theme"], width="stretch")

        st.markdown("#### Todas as conexoes")
        st.dataframe(results["graph"], width="stretch")


def render_tech_log() -> None:
    if st.session_state.last_tech_log is not None:
        st.markdown("### Log tecnico da ultima execucao")
        st.json(st.session_state.last_tech_log)


def render_app() -> None:
    tab_exp, tab_search, tab_import = st.tabs(["Experimentos", "Busca Grafo", "Importacao Textos"])
    with tab_exp:
        render_experiments_tab()
    with tab_search:
        render_search_tab()
    with tab_import:
        render_import_tab()
    render_tech_log()


def run() -> None:
    st.set_page_config(page_title="Doutorado Extrator Grafos", layout="wide")
    st.title("Doutorado Extrator Grafos - Knowledge Graph Lab")
    st.caption("A tela inicial nao carrega dados automaticamente. Use os botoes para processar ou buscar.")
    init_session_state()

    try:
        render_app()
    except Exception as exc:
        st.error(f"Erro ao renderizar a aplicacao: {exc}")
        with st.expander("Detalhes tecnicos"):
            st.code(traceback.format_exc())


if __name__ == "__main__":
    run()
