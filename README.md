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

# 🗺️ System Architecture

```text
              ┌────────────────────────────┐
              │  User Business Question    │
              └─────────────┬──────────────┘
                            │
                            ▼
        ┌──────────────────────────────────────┐
        │ Business Glossary & Semantic Layer  │
        │ KPI definitions / business rules    │
        └─────────────┬────────────────────────┘
                      │
                      ▼
        ┌──────────────────────────────────────┐
        │      RAG Context Retrieval Layer     │
        │ schema / glossary / marts / examples │
        └─────────────┬────────────────────────┘
                      │
                      ▼
        ┌──────────────────────────────────────┐
        │      LLM Text-to-SQL Generator       │
        └─────────────┬────────────────────────┘
                      │
                      ▼
        ┌──────────────────────────────────────┐
        │        SQL Execution Engine          │
        └─────────────┬────────────────────────┘
                      │
                      ▼
        ┌──────────────────────────────────────┐
        │     SQL Evaluator & Validation       │
        │ expected vs generated SQL checking   │
        └─────────────┬────────────────────────┘
                      │
                      ▼
        ┌──────────────────────────────────────┐
        │      Structured Analytics Output     │
        └──────────────────────────────────────┘
```

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

Advanced vector-based retrieval and document retrieval capabilities are planned in future development phases.

This architecture improves:

* business consistency
* KPI reliability
* SQL accuracy
* hallucination prevention
* enterprise governance alignment

---

# 📂 Repository Structure

```text
ai-analytics-copilot/
│
├── config/
│   ├── glossary.json
│   └── settings.yaml
│
├── data/
│   ├── database/
│   │   └── thelook_ecommerce.db
│   │
│   ├── documents/
│   │   └── market_reports/
│   │
│   ├── vectorstore/
│   │
│   └── evaluation/
│       └── test_cases.json
│
├── notebooks/
│   ├── 01_data_wrangling.ipynb
│   ├── 02_load_clean_to_sqlite.ipynb
│   ├── 03_create_marts.ipynb
│   ├── 04_data_mart_validation.ipynb
│   └── 05_sql_evaluator_prototype.ipynb
│
├── outputs/
│   ├── sql_evaluation_result.csv
│   └── sql_evaluation_summary.json
│
├── src/
│   ├── router.py
│   ├── text_to_sql.py
│   ├── advanced_rag.py
│   ├── sql_evaluator.py
│   ├── merge_layer.py
│   └── utils.py
│
├── app/
│   └── main.py
│
├── README.md
└── requirements.txt
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

* Ground truth SQL benchmarking
* Result-based SQL validation
* Shape / column / value comparison
* Pass / fail scoring
* Benchmark accuracy reporting
* JSON / CSV evaluation export
* SQL execution error tracing
* Semantic business logic validation
* Required SQL pattern checking (in progress)

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

* KPI aggregation
* ranking queries
* time filtering
* segmentation logic
* product analytics
* customer analytics
* return analysis
* profitability analysis

Each benchmark case may contain:

* natural language question
* business metric mapping
* expected SQL
* required SQL checks
* generated SQL
* evaluation results

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

## 1. Clone Repository

```bash
git clone https://github.com/yourusername/ai-analytics-copilot.git
cd ai-analytics-copilot
```

---

## 2. Create Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 3. Install Requirements

```bash
pip install -r requirements.txt
```

---

## 4. Launch Notebook Environment

```bash
jupyter notebook
```

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

The evaluator currently supports:

* SQL execution benchmarking
* batch SQL evaluation
* benchmark pass/fail reporting
* JSON/CSV export
* semantic SQL validation

---

# 🚀 Future Improvements

* OpenAI function-calling router
* dynamic schema retrieval
* SQL confidence scoring
* semantic KPI matching
* ambiguity detection
* production SQL sandboxing
* Streamlit analytics dashboard
* automated RAG evaluation
* LLM-generated SQL benchmarking
* hybrid SQL + document reasoning

---

# 📜 License

This project is licensed under the MIT License.