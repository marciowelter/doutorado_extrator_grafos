from __future__ import annotations

import json
import time
import traceback
from datetime import datetime, timezone

import streamlit as st

from src.application.pipeline_use_case import PipelineUseCase
from src.application.search_use_case import SearchUseCase
from src.infrastructure.llm.llamaindex_client import get_last_extraction_debug


RECOMMENDED_MAX_CHARS = 3000
HARD_WARN_CHARS = 7000


def init_session_state() -> None:
    if "last_extraction" not in st.session_state:
        st.session_state.last_extraction = None

    if "last_search_results" not in st.session_state:
        st.session_state.last_search_results = None

    if "last_processing_info" not in st.session_state:
        st.session_state.last_processing_info = None

    if "last_tech_log" not in st.session_state:
        st.session_state.last_tech_log = None


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
    st.subheader("Busca vetorial + grafo")
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
                status_box.write("1/2 Consultando indice vetorial")
                results = search_sync(keyword)
                status_box.write("2/2 Carregando conexoes de grafo")
                status_box.update(label="Busca concluida", state="complete")
                st.session_state.last_search_results = results
                log_exec("ok", "Busca", started, {"keyword": keyword})
            except Exception as exc:
                status_box.update(label="Falha na busca", state="error")
                log_exec("error", "Busca", started, {"keyword": keyword}, str(exc))
                st.error(f"Falha na busca: {exc}")

    render_search_results(st.session_state.last_search_results)


def render_search_results(results: dict[str, list[dict[str, str]]] | None) -> None:
    if results is None:
        st.info("Nenhum dado carregado ainda. Informe uma palavra-chave e clique em Buscar.")
        return

    st.markdown("### Trechos correlacionados (pgvector)")
    if not results["vector"]:
        st.info("Nenhum trecho encontrado.")
    else:
        for item in results["vector"]:
            st.write(item["content"])

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
    tab_exp, tab_search = st.tabs(["Experimentos", "Busca Hibrida"])
    with tab_exp:
        render_experiments_tab()
    with tab_search:
        render_search_tab()
    render_tech_log()


def run() -> None:
    st.set_page_config(page_title="llamaindex KG", layout="wide")
    st.title("LlamaIndex Knowledge Graph Lab")
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
