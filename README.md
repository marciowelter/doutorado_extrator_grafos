# Doutorado Extrator Grafos

Pipeline de extração de conhecimento para textos legislativos em português brasileiro. O projeto lê trechos do datamart relacional, extrai **entidades nomeadas** (GLiNER) e **temas** (BERTopic), persiste um grafo de conhecimento no **Apache AGE** (PostgreSQL) e oferece jobs batch via CLI para importação, normalização e correção de labels.

## Visão geral

```
Datamart PostgreSQL (schema doutorado)
        │
        ▼
ImportTextsUseCase / PipelineUseCase
        │
        ├─ GLiNER ──────────► entidades (PESSOA, ORGANIZACAO, LOCAL, …)
        ├─ BERTopic ────────► temas (SAUDE, EDUCACAO, …)
        └─ relacionamentos fixos ENTIDADE → TEMA (RELACIONA)
        │
        ▼
Apache AGE (grafo doutorado_graph, nós :Entidade)
        │
        ├─ normalize-graph ─► unifica nós duplicados por nome
        └─ correct-labels ──► corrige label TEMA/ENTIDADE via LLM
```

### Modelo de dados no grafo

Todos os nós usam o label Apache AGE `:Entidade`. A distinção semântica fica em:

| Campo | Valores | Descrição |
|-------|---------|-----------|
| `label` | `TEMA`, `ENTIDADE` | Tipo do nó |
| `properties.categoria` | `TEMA`, `PESSOA`, `ORGANIZACAO`, `LOCAL`, … | Categoria fina (GLiNER ou `TEMA`) |
| `properties.contexto` | texto | Trecho de contexto da extração |
| `name` | string normalizada | Identificador do nó |

Tabela física AGE: `{AGE_GRAPH_NAME}."Entidade"` (ex.: `doutorado_graph."Entidade"`).

## Estrutura do projeto

```
config/settings.py          # Variáveis de ambiente
main.py                     # CLI batch
src/
  application/              # Casos de uso
    import_texts_use_case.py
    correct_labels_use_case.py
    pipeline_use_case.py
    search_use_case.py
  domain/                   # Modelos e interfaces
  infrastructure/
    database/               # PostgreSQL + Apache AGE
    llm/                    # GLiNER, BERTopic, classificadores LLM
  ui/app.py                 # Interface Streamlit
tests/
requirements.txt
.env.example
```

## Pré-requisitos

- Python 3.12+
- PostgreSQL com extensão **Apache AGE**
- Acesso ao banco `banco` (datamart + grafo)
- **Ollama** (opcional, classificação local) ou **Gemini API** (recomendado para correção de labels)

## Instalação

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Edite .env com credenciais e URLs
```

### Variáveis de ambiente principais

| Variável | Descrição |
|----------|-----------|
| `POSTGRES_*` | Conexão PostgreSQL |
| `POSTGRES_SCHEMA` | Schema relacional (`doutorado`) |
| `AGE_GRAPH_NAME` | Nome do grafo AGE (`doutorado_graph`) |
| `OLLAMA_*` | URL, modelo e timeout do Ollama |
| `GEMINI_API_KEY` | Chave da API Google Gemini |
| `GEMINI_MODEL` | Modelo Gemini (padrão: `gemini-2.5-flash`) |
| `GLINER_*` | Modelo e labels do extrator de entidades |

Consulte `.env.example` para a lista completa.

## Interface Streamlit

```bash
streamlit run src/ui/app.py
```

Permite extrair texto avulso, buscar no grafo e disparar importação de textos pendentes pela interface.

---

## Processos batch (CLI)

Todos os jobs batch são executados via `main.py`:

```bash
.venv/bin/python main.py [opções]
```

A saída padrão é **JSON** (stdout), ideal para logs estruturados e automação.

### 1. Importação de textos (`--import-texts`)

Processa trechos pendentes do datamart (`trecho.grafo = false`, texto > 1000 caracteres), extrai entidades/temas e grava no grafo. Marca cada trecho como processado.

**Comando:**

```bash
.venv/bin/python main.py --import-texts
```

**Parâmetros:**

| Parâmetro | Padrão | Descrição |
|-----------|--------|-----------|
| `--import-texts` | — | Ativa o job |
| `--progress-batch-size` | `25` | Frequência do log de progresso (a cada N registros) |
| `--background` | — | Executa em background (desacoplado do terminal) |
| `--background-log` | `import_texts_background.log` | Arquivo de log no modo background |

**Exemplo em background:**

```bash
.venv/bin/python main.py --import-texts --background --background-log logs/import.log
tail -f logs/import.log
```

**Resumo JSON retornado:**

```json
{
  "total": 150,
  "attempted": 150,
  "successful": 148,
  "failed": 2,
  "elapsed_seconds": 3600.5,
  "average_record_seconds": 24.0
}
```

**Eventos de log (stdout):** `batch_progress`, `retry`, `record_failure`.

---

### 2. Correção de labels TEMA/ENTIDADE (`--correct-labels`)

Classifica cada nó `:Entidade` via LLM (Gemini ou Ollama) e corrige o campo `label` quando a classificação difere do valor atual. Gera log JSONL detalhado por nó.

**Recomendação:** use `--correct-labels-provider gemini` em produção. O Ollama local (`qwen2.5:0.5b`) tende a errar em temas abstratos.

#### Modos de seleção de nós

| Modo | Flags | Descrição |
|------|-------|-----------|
| Sequencial | `--correct-labels-limit N [--correct-labels-offset M]` | N primeiros nós (ordenados por id), com deslocamento |
| Aleatório | `--correct-labels-random --correct-labels-limit N [--correct-labels-exclude-first K]` | N nós sorteados, opcionalmente excluindo os K primeiros |
| **Base inteira** | `--correct-labels-all` | **Todos** os nós `Entidade` do grafo |

#### Parâmetros completos

| Parâmetro | Padrão | Descrição |
|-----------|--------|-----------|
| `--correct-labels` | — | Ativa o job |
| `--correct-labels-provider` | `gemini` | `gemini` ou `ollama` |
| `--correct-labels-limit` | `100` | Quantidade de nós (ignorado com `--correct-labels-all`) |
| `--correct-labels-offset` | `0` | Deslocamento sequencial (incompatível com `--correct-labels-all`) |
| `--correct-labels-all` | — | Processa **toda** a base de Entidades |
| `--correct-labels-random` | — | Amostragem aleatória (incompatível com `--correct-labels-all`) |
| `--correct-labels-exclude-first` | `0` | Exclui os N primeiros ids (somente com `--random`) |
| `--correct-labels-dry-run` | — | Analisa e registra no log **sem gravar** no banco |
| `--correct-labels-log` | `/tmp/doutorado_label_correction.log` | Arquivo de log JSONL |
| `--background` | — | Executa em background (útil com `--correct-labels-all`) |

#### Exemplos

```bash
# Piloto: 100 primeiros, dry-run, Gemini
.venv/bin/python main.py \
  --correct-labels \
  --correct-labels-provider gemini \
  --correct-labels-limit 100 \
  --correct-labels-dry-run \
  --correct-labels-log /tmp/label_pilot.log

# Amostra aleatória de 200 nós (excluindo os 100 primeiros), dry-run
.venv/bin/python main.py \
  --correct-labels \
  --correct-labels-provider gemini \
  --correct-labels-random \
  --correct-labels-limit 200 \
  --correct-labels-exclude-first 100 \
  --correct-labels-dry-run \
  --correct-labels-log /tmp/label_random200.log

# Base inteira, dry-run (pode levar horas)
.venv/bin/python main.py \
  --correct-labels \
  --correct-labels-provider gemini \
  --correct-labels-all \
  --correct-labels-dry-run \
  --correct-labels-log /tmp/label_full_dryrun.log

# Base inteira em background, aplicando correções
.venv/bin/python main.py \
  --correct-labels \
  --correct-labels-provider gemini \
  --correct-labels-all \
  --correct-labels-log logs/label_full_apply.log \
  --background

tail -f logs/label_full_apply.log
```

#### Formato do log JSONL

Cada linha é um objeto JSON:

- **`batch_started`** — metadados: `provider`, `limit`, `process_entire_graph`, `total_entities_in_graph`, `dry_run`
- **Registro por nó** — `node_id`, `name`, `current_label`, `predicted_label`, `would_update`, `updated`, `justificativa`, `error`
- **`batch_finished`** — totais: `processed`, `updated`, `would_update`, `unchanged`, `errors`

Filtrar sugestões de correção:

```bash
grep '"would_update": true' /tmp/label_pilot.log
grep '"error":' /tmp/label_pilot.log | grep -v '"error": null'
```

---

### 3. Normalização do grafo (`--normalize-graph`)

Unifica nós `:Entidade` com o mesmo `name` normalizado, remove duplicatas e reconstrói arestas. Normaliza `label` e `properties.categoria` nos nós canônicos.

**Comando:**

```bash
.venv/bin/python main.py --normalize-graph
```

**Resumo JSON:**

```json
{
  "nodes_before": 5000,
  "nodes_after": 4200,
  "duplicates_removed": 800,
  "relationships_before": 12000,
  "relationships_after": 11500
}
```

> Este job altera o grafo diretamente. Não possui modo dry-run.

---

### 4. Extração avulsa (`--text`)

Processa um único texto (útil para testes). Não é batch.

```bash
.venv/bin/python main.py --text "Texto do discurso ou trecho aqui..."
```

Retorna JSON com `entities` e `relationships` extraídos (e persiste no grafo).

---

## Fluxo recomendado de operação

1. **Importar textos pendentes:** `--import-texts` (em background se volume alto)
2. **Normalizar duplicatas:** `--normalize-graph`
3. **Validar labels (dry-run):** `--correct-labels --correct-labels-dry-run` em amostra
4. **Revisar log JSONL** e decidir aplicação
5. **Aplicar correção:** `--correct-labels-all` (ou lotes sequenciais) sem `--dry-run`

## Testes

```bash
.venv/bin/python -m pytest tests/
```

## Tecnologias

| Componente | Uso |
|------------|-----|
| PostgreSQL + Apache AGE | Grafo de conhecimento |
| psycopg3 | Driver de banco |
| GLiNER | Extração de entidades |
| BERTopic | Extração de temas |
| Google Gemini / Ollama | Classificação TEMA vs ENTIDADE |
| Pydantic | Validação de modelos |
| Streamlit | Interface web |
