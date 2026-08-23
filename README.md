# AI Analytics Copilot with Governed Text-to-SQL & Advanced RAG

An enterprise-style AI analytics copilot prototype designed to bridge the gap between natural business language and technical analytics systems.

This project combines:

* Governed Text-to-SQL generation
* Business glossary semantic mapping
* Advanced RAG retrieval
* SQL evaluation benchmarking
* Structured analytics marts
* AI validation pipelines

The system is designed to simulate how modern enterprise AI analytics assistants (e.g., Databricks Genie, Snowflake Cortex Analyst, Microsoft Fabric Copilot, ThoughtSpot Sage) operate in production environments.

---

# 🎬 Demo

![The copilot answering "How does our return rate compare to the industry?"](docs/demo.gif)

One question, both branches: the router classifies it as `both`, the SQL branch
computes the order-level return rate from `mart_order_summary`, the RAG branch
retrieves the industry figure from the report corpus, and the merge step writes
the PM report. The generated SQL and the retrieved pages are one click away
under each answer.

---

# 🗺️ System Architecture

```text
                    ┌────────────────────────────┐
                    │   User Business Question   │
                    └─────────────┬──────────────┘
                                  │
                                  ▼
                    ┌────────────────────────────┐
                    │      classify (LLM)        │
                    │ sql_only / rag_only / both │
                    └─────────────┬──────────────┘
                                  │
                 ┌────────────────┴────────────────┐
                 │        (run concurrently)       │
                 ▼                                 ▼
   ┌──────────────────────────┐      ┌──────────────────────────┐
   │   Governed Text-to-SQL   │      │      Advanced RAG        │
   │ glossary retrieval       │      │ hybrid BM25 + vector     │
   │ → prompt → LLM → SQL     │      │ → RRF fusion → cited     │
   │ → SQLite execution       │      │   answer over 12 PDFs    │
   └────────────┬─────────────┘      └────────────┬─────────────┘
                │  DataFrame + SQL                │  answer + source chunks
                └────────────────┬────────────────┘
                                 ▼
                   ┌───────────────────────────┐
                   │    merge (LLM → JSON)     │
                   │ PM report, no fabrication │
                   └─────────────┬─────────────┘
                                 ▼
                   ┌───────────────────────────┐
                   │  Streamlit chat + charts  │
                   │  SQL / data / sources     │
                   └───────────────────────────┘
```

Shared conversation memory (`src/rag/memory.py`) feeds the classifier, the RAG
chain, and the merge step, so follow-up questions resolve against earlier turns.

A separate evaluation path benchmarks the SQL branch on its own — see
[SQL Evaluation Framework](#-sql-evaluation-framework).

---

# 🧠 Governed AI Analytics Layer

The system integrates a semantic governance layer to improve SQL reliability and reduce hallucinations.

Instead of directly generating SQL from raw database schemas, the pipeline retrieves:

* KPI definitions
* approved business logic
* preferred marts/views
* metric constraints
* semantic business rules
* schema relationships

before prompting the LLM.

Current implementation includes:

- KPI glossary retrieval
- business metric mapping
- preferred source selection
- time period identification
- prompt construction
- SQL execution validation

Vector-based document retrieval is implemented and in use: the RAG branch runs
hybrid BM25 + vector search with RRF fusion over 12 annual and industry report
PDFs (`src/rag/`), and the router runs it alongside the SQL branch.

This architecture improves:

* business consistency
* KPI reliability
* SQL accuracy
* hallucination prevention
* enterprise governance alignment

---

# 📂 Repository Structure

```text
AI-Analytics-Chatbot/
│
├── app/
│   ├── dashboard.py            # Streamlit chat UI (primary entry point)
│   ├── charts.py               # chart selection for SQL results (pure, tested)
│   ├── app.py                  # SQL-only CLI loop
│   └── copilot.py              # unified copilot CLI (router → SQL + RAG → PM report)
│
├── src/
│   ├── llm.py                  # shared HuggingFace chat + embedding providers
│   ├── router.py               # route classification, parallel execution, merge
│   │
│   ├── sql_agent/
│   │   ├── config.py           # canonical paths (glossary, database, test cases)
│   │   ├── retriever.py        # glossary metric retrieval + live schema lookup
│   │   ├── prompt_builder.py   # metric/schema/time-rule prompt assembly
│   │   ├── sql_generator.py    # LLM call + SQL extraction
│   │   ├── sql_executor.py     # SQLite execution
│   │   ├── sql_pipeline.py     # retrieve → prompt → generate → execute
│   │   ├── sql_evaluator.py    # result comparison
│   │   └── accuracy_report.py  # end-to-end benchmark runner
│   │
│   └── rag/
│       ├── config.py           # models, chunk sizes, paths
│       ├── parsing.py          # PDF parsing + metadata tagging
│       ├── vectorstore.py      # parent/child chunking, Chroma
│       ├── retrieval.py        # hybrid BM25 + vector with RRF fusion
│       ├── chain.py            # RAG prompt + chain assembly
│       ├── memory.py           # conversation window + SQLite persistence
│       ├── clarification.py    # vague-question detection
│       ├── conversation.py     # multi-turn loop
│       └── main.py             # RAG-only demo
│
├── config/
│   └── glossary.json           # KPI definitions, formulas, business rules
│
├── data/
│   ├── database/               # thelook_ecommerce.db (built, gitignored)
│   ├── reports/                # 12 annual + industry PDFs
│   ├── evaluation/
│   │   └── test_cases.json     # benchmark questions + expected SQL
│   └── processed/              # sessions.db (created at runtime)
│
├── scripts/
│   └── build_db.py             # reproducible database builder
│
├── tests/
│   ├── test_extract_sql.py
│   ├── test_sql_evaluator.py
│   ├── test_charts.py
│   ├── test_chain.py           # RAG context formatting / citations
│   ├── test_rag_parsing.py     # year + doc_type from filenames
│   ├── test_rag_retrieval.py   # RRF fusion + metadata filtering
│   └── test_glossary_schema.py # glossary ↔ live schema consistency
│
├── .github/workflows/tests.yml # pytest on push and PR
│
├── notebooks/                  # historical exploration (01–05)
├── outputs/                    # evaluation reports
│
├── requirements.txt
├── requirements-dev.txt
├── .python-version
└── README.md
```

---

# 🏗️ Data Warehouse & Analytics Marts

The project uses SQLite as the analytical warehouse layer.

Structured marts were designed to support downstream AI analytics querying and KPI standardization.

## Implemented Analytics Marts

| Mart | Purpose |
|--------|---------|
| mart_daily_sales | Daily sales and revenue reporting |
| mart_order_summary | Order-level profitability and fulfillment metrics |
| mart_product_sales | Product and category performance analytics |
| mart_user_summary | Customer purchasing behaviour |
| mart_user_segment | Customer lifecycle segmentation |

---

# 📊 SQL Evaluation Framework

A custom SQL evaluation framework was developed to benchmark and validate AI-generated SQL queries against predefined business ground truth logic.

## Current Evaluation Features

* Ground truth SQL benchmarking (`src/sql_agent/accuracy_report.py`)
* Result-based validation: equivalent SQL passes, identical text is not required
* Shape check, then per-column comparison — numeric with tolerance, text exact
* Pass / fail scoring with execution-error tracing
* CSV report written to `outputs/`
* Glossary ↔ live schema consistency checks (`tests/test_glossary_schema.py`)

Semantic business-rule validation was prototyped and removed; it is listed
under Future Improvements rather than described as working.

---

# 🔍 Evaluation Pipeline

```text
Question
↓
Expected SQL
↓
Generated SQL
↓
SQL Execution
↓
Result Comparison
↓
Pass / Fail Report
```

---

# 📈 Benchmark & Ground Truth Testing

A benchmark dataset was manually created to evaluate Text-to-SQL performance across multiple analytics scenarios.

## Coverage Areas

Five cases in `data/evaluation/test_cases.json`, covering KPI aggregation
(revenue, AOV), ranking with grouping (top categories), time filtering,
customer activity windows, and return analysis.

Each case contains:

* natural language question
* expected SQL, derived from `config/glossary.json` and tagged with
  `expected_sql_source`

The runner executes the expected and generated SQL against the same database
and compares results, so it measures whether the pipeline implements the
glossary — not whether the glossary is business-correct.

---

# 🛠️ Key Technical Features

## 1. Governed Text-to-SQL

Business glossary mappings are injected into prompts to align AI-generated SQL with enterprise KPI definitions.

---

## 2. Advanced RAG Retrieval

Hybrid retrieval pipelines provide:

* schema context
* glossary definitions
* business rules
* example SQL patterns
* document intelligence

before SQL generation.

---

## 3. SQL Evaluator Engine

AI-generated SQL is validated against predefined ground truth outputs to detect:

* incorrect aggregations
* missing filters
* incorrect joins
* semantic KPI mismatches
* hallucinated SQL logic

---

## 4. Semantic Governance Layer

The system distinguishes between:

* officially defined KPIs
* inferred metrics
* ambiguous metrics
* unsupported business questions

to reduce unreliable analytics generation.

---

## 5. Structured Analytics Layer

Analytics marts and KPI standardization were designed to improve:

* SQL consistency
* query reliability
* AI prompt quality
* business semantic alignment

---

# 🚀 Getting Started

## 1. Requirements

Python **3.11–3.13**. Not 3.14: every published version of `unstructured`,
which parses the report PDFs, requires `<3.14`. `.python-version` pins 3.13 for
`uv` and `pyenv` users.

```bash
git clone https://github.com/ChoWingDev/AI-Analytics-Chatbot.git
cd AI-Analytics-Chatbot
```

---

## 2. Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## 3. Credentials

```bash
cp .env.example .env
```

Then fill in:

| variable | where to get it | needed for |
|---|---|---|
| `HF_TOKEN` | [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) (Read scope) | every LLM call |
| `KAGGLE_USERNAME`, `KAGGLE_KEY` | [kaggle.com/settings/account](https://www.kaggle.com/settings/account) → Create New Token | building the database |

Kaggle credentials are only needed for the download; `build_db.py --csv-dir`
accepts CSVs you already have.

---

## 4. Build the database

```bash
python scripts/build_db.py
```

Downloads TheLook, loads 7 base tables (~2.4M events rows), and builds the 5
analytics marts. Produces roughly 528 MB at
`data/database/thelook_ecommerce.db`. Takes a few minutes.

---

## 5. Run

```bash
streamlit run app/dashboard.py   # the dashboard — start here
```

Then open http://localhost:8501. Command-line entry points still work:

```bash
python -m app.copilot     # unified copilot: internal KPIs + industry reports
python -m app.app         # SQL-only loop
python -m src.rag.main    # RAG-only demo
```

The first run that touches RAG parses all 12 PDFs and embeds ~105k chunks on
CPU — around 15 minutes. Results are cached to `data/parsed_docs.pkl` and
`chroma_db/`, so later runs start in seconds.

---

## 6. Tests and benchmarks

```bash
pip install -r requirements-dev.txt
pytest                                    # unit tests
python -m src.sql_agent.accuracy_report   # end-to-end Text-to-SQL benchmark
```

`tests/test_glossary_schema.py` checks every glossary metric against the live
database schema, and skips if the database has not been built.

---

## ⚠️ A note on the data

`scripts/build_db.py` pulls the Kaggle mirror
`mustafakeser4/looker-ecommerce-bigquery-dataset`. TheLook is synthetic and
regenerated over time, so each mirror is a different snapshot — **this is not
the same data the project's original numbers came from.** The same query for
January 2021 category revenue returns 4,355.92 for Outerwear & Coats today
versus 7,506.91 originally, and the difference is not a constant factor.

Any figure quoted in this README, the notebooks, or the debugging log below
predates the current snapshot. `outputs/sql_evaluation_result.csv` and
`src/sql_agent/sql_accuracy_report.csv` are kept in the repo as the record of
the original dataset's values.

The benchmark in `data/evaluation/test_cases.json` is unaffected: it executes
expected and generated SQL against the *same* database and compares results,
so it measures whether the pipeline implements `config/glossary.json` — not
whether the glossary is business-correct.

---

# 🛠️ Debugging Log — AI Analytics Copilot (Text-to-SQL Pipeline)

## Overview

This document records the key issues encountered and resolved during development of the Text-to-SQL RAG pipeline, including root cause analysis and solutions applied.

---

## Issue 1: SQL Generator Ignoring RAG Pipeline

**Symptom**
The `sql_generator.py` test was passing the raw user question directly to the LLM, bypassing the entire RAG pipeline. The LLM responded with general world knowledge instead of querying the local database.

**Root Cause**
The `__main__` block in `sql_generator.py` called `generate_sql(question)` directly without first running retrieval and prompt building.

**Solution**
Wire the full pipeline in the correct order:
1. `GlossaryRetriever.retrieve(question)` — fetch metrics, schema, time periods
2. `PromptBuilder.build_prompt(question, retrieval_result)` — build enriched prompt
3. `SQLGenerator.generate_sql(prompt)` — pass the enriched prompt, not the raw question

---

## Issue 2: Deprecated `google.generativeai` Package

**Symptom**
`FutureWarning` on every run stating that `google.generativeai` is no longer supported.

**Solution**
Migrate to the new `google.genai` package. The API interface changed significantly:

```python
# Old
import google.generativeai as genai
genai.configure(api_key=KEY)
model = genai.GenerativeModel("models/gemini-2.5-flash")
response = model.generate_content(prompt)

# New
from google import genai
client = genai.Client(api_key=KEY)
response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=prompt
)
```

---

## Issue 3: Inconsistent Query Results Between `mart_product_sales` and Raw Table Join

**Symptom**
Two queries for the same question ("top 5 product categories by revenue in 202101") returned different numbers depending on whether `mart_product_sales` or a raw `order_items JOIN products` was used.

**Root Cause**
The two approaches used different date source columns:
- `mart_product_sales.order_date` = `DATE(orders.created_at)`
- Raw query used `DATE(order_items.created_at)`

Although these timestamps should theoretically be the same, ~2.46% of `order_items.created_at` records are missing or have slight timestamp differences, causing records to fall into different date buckets after `DATE()` truncation.

**Solution**
- Standardise on `mart_product_sales` as the single source of truth for product/category revenue queries
- Added `category_revenue` as a separate glossary metric explicitly pointing to `mart_product_sales`
- Added business logic rule: *"Date filters must be applied to `order_date` which represents `DATE(orders.created_at)`"*

---

## Issue 4: Prompt Too Long — Hitting API Quota

**Symptom**
Prompts were excessively long due to RAG chunks from `thelook_db_documentation.docx` injecting redundant and irrelevant schema information. This caused rapid exhaustion of the free-tier API quota (20 requests/day).

**Root Cause**
- `RecursiveCharacterTextSplitter` with `chunk_size=900` was blindly splitting the documentation, causing chunks to mix content from multiple tables
- Both RAG doc chunks and SQLite PRAGMA were injecting schema information, resulting in duplication

**Solution**
Replaced the RAG doc approach with a leaner two-source architecture:

| Source | Purpose |
|---|---|
| `glossary.json` (structured JSON) | Metric definitions, formulas, business logic, preferred source |
| SQLite `PRAGMA table_info()` | Exact column list per table, dynamically fetched |

Key changes:
- Removed `build_schema_documents()` and Word doc embedding entirely
- Compressed schema injection to a single line per table: `Table mart_product_sales (order_date, category, gross_sales, ...)`
- Capped metric injection to top 2 most relevant metrics
- Capped business logic rules to 3 per metric

---
## Final Results

| Metric | Value |
|---|---|
| Total Test Cases | 5 |
| Passed | 4 |
| Execution Accuracy | **80%** |
| Remaining Failure | `top 5 product categories by revenue in 202101` — keyword routing fix pending validation |

---

## Key Takeaways

> **Glossary quality is the single most important factor in Text-to-SQL accuracy.**

The priority order for context injection is:

1. **Glossary** — controls which table, field, and formula the LLM uses
2. **Prompt instructions** — enforces constraints and prevents hallucination
3. **SQLite PRAGMA schema** — validates column existence
4. **RAG doc** — only needed for edge cases not covered by the glossary

# 📊 Current Evaluation Results

Text-to-SQL benchmark: **5 / 5** cases pass against the current snapshot
(`python -m src.sql_agent.accuracy_report`).

Unit tests: **74 passing** — SQL extraction, result comparison, glossary /
schema consistency, RAG parsing, rank fusion, retrieval filtering, context
formatting, and chart selection. They run in CI on every push with no database,
no vector store, and no API token.

Read the benchmark number with its caveat: expected SQL is derived from the
glossary the pipeline also reads, so a passing score means the pipeline
faithfully implements the glossary. It is a consistency measure, not an
independent accuracy measure.

---

# ⚖️ What works, and what does not

**Works**

* End-to-end demo in the browser: routing, parallel SQL + RAG, merged PM report,
  charts, and the SQL and source pages behind every answer
* 5/5 on the Text-to-SQL benchmark, 74 unit tests, CI green on push
* Answers cite source document, page, year, and document type
* The merge prompt refuses to invent numbers: a failed branch becomes "N/A"
  rather than a plausible fabrication, and the UI shows the failure

**Does not work, or is not what it looks like**

* **The 5/5 benchmark is a consistency measure.** Expected SQL is derived from
  the same `config/glossary.json` the pipeline reads. It proves the pipeline
  implements the glossary; it cannot detect a glossary that is business-wrong.
* **The merge step compares metrics without checking their grain.** The return
  rate here is order-level (10.01% — the share of orders containing a return).
  Published industry return rates are usually unit-level (the share of items
  returned). Nothing in the pipeline knows the difference, so when both numbers
  are present it will label the gap "Above" or "Below average" on a comparison
  that is not like-for-like. Check both grains yourself before believing a
  status. (When no industry figure is retrieved the report says "N/A" rather
  than inventing one — that guard works; grain awareness is the missing part.)
* **Three glossary metrics are unaudited.** `aov`, `active_users` and
  `conversion_rate` pass the schema tests, but their business logic has not had
  the review `return_rate` received. `conversion_rate` queries the raw `events`
  table because no funnel mart exists.
* **Single user.** The dashboard caches one agent per server process and that
  agent holds one conversation memory. Concurrent users would interleave.
* **Not deployed.** ~898 MB of runtime artifacts (528 MB database, 354 MB
  Chroma, 17 MB parse cache) and 8-13s per answer make free hosting a poor
  demo. It runs locally.
* **Numbers predate the current data snapshot** wherever they appear in the
  notebooks and the debugging log — see the note on the data above.

---

# 🚀 Future Improvements

* OpenAI function-calling router
* dynamic schema retrieval
* SQL confidence scoring
* semantic KPI matching
* ambiguity detection
* production SQL sandboxing
* semantic business-rule validation (prototyped, then removed)
* automated RAG evaluation
* LLM-generated SQL benchmarking
* grain-aware metric comparison in the merge step
* audit of the remaining glossary metrics
* hybrid SQL + document reasoning

---

# 📜 License

This project is licensed under the MIT License.