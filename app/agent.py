"""
src/sql_agent/agent.py
----------------------
Person A — Week 1–2
Text-to-SQL agent using LangChain + TheLook SQLite database.
"""

import json
import re
import sqlite3
from pathlib import Path

from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI

from src.glossary.loader import get_glossary_prompt

DB_PATH = Path("data/processed/thelook.db")

FEW_SHOT_EXAMPLES = [
    {
        "question": "What was the churn rate for email campaigns last month?",
        "sql": """
            WITH email_users AS (
                SELECT DISTINCT user_id FROM events
                WHERE traffic_source = 'Email'
                AND DATE(created_at) >= DATE('now', 'start of month', '-1 month')
                AND DATE(created_at) < DATE('now', 'start of month')
            ),
            churned AS (
                SELECT DISTINCT user_id FROM orders
                WHERE status = 'Cancelled'
                AND user_id IN (SELECT user_id FROM email_users)
            )
            SELECT
                ROUND(COUNT(DISTINCT c.user_id) * 100.0 / COUNT(DISTINCT e.user_id), 2) AS churn_rate_pct
            FROM email_users e
            LEFT JOIN churned c ON e.user_id = c.user_id
        """,
    },
    {
        "question": "What is the ROI for each traffic source last month?",
        "sql": """
            SELECT
                u.traffic_source,
                ROUND(((SUM(oi.sale_price) - SUM(oi.cost)) / SUM(oi.cost)) * 100, 2) AS roi_pct,
                ROUND(SUM(oi.sale_price), 2) AS total_revenue,
                ROUND(SUM(oi.cost), 2) AS total_cost
            FROM order_items oi
            JOIN orders o ON oi.order_id = o.order_id
            JOIN users u ON oi.user_id = u.id
            WHERE o.status = 'Complete'
            AND DATE(o.created_at) >= DATE('now', 'start of month', '-1 month')
            AND DATE(o.created_at) < DATE('now', 'start of month')
            GROUP BY u.traffic_source
            ORDER BY roi_pct DESC
        """,
    },
]


def build_few_shot_text() -> str:
    lines = ["\n## Example SQL Queries\n"]
    for i, ex in enumerate(FEW_SHOT_EXAMPLES, 1):
        lines.append(f"Example {i}:")
        lines.append(f"Question: {ex['question']}")
        lines.append(f"SQL: {ex['sql'].strip()}\n")
    return "\n".join(lines)


FORBIDDEN_KEYWORDS = ["DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "TRUNCATE"]


def validate_sql(sql: str) -> bool:
    """Reject any SQL that tries to mutate the database."""
    upper = sql.upper()
    for kw in FORBIDDEN_KEYWORDS:
        if re.search(rf"\b{kw}\b", upper):
            return False
    return True


def build_system_prompt() -> str:
    glossary = get_glossary_prompt()
    few_shots = build_few_shot_text()
    return f"""You are a senior data analyst helping a Product Manager query an eCommerce database.
Always generate safe, read-only SQL. Never use DROP, DELETE, UPDATE, INSERT, or ALTER.
Return clear, concise answers with the actual numbers.

{glossary}

{few_shots}

Database: TheLook eCommerce (SQLite)
Tables: users, orders, order_items, products, events
"""


def create_sql_agent(db_path: Path = DB_PATH) -> AgentExecutor:
    db = SQLDatabase.from_uri(f"sqlite:///{db_path}")
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)

    prompt = ChatPromptTemplate.from_messages([
        ("system", build_system_prompt()),
        MessagesPlaceholder("chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ])

    agent = create_openai_tools_agent(llm=llm, tools=toolkit.get_tools(), prompt=prompt)
    return AgentExecutor(agent=agent, tools=toolkit.get_tools(), verbose=True)


async def run_sql_query(question: str, agent: AgentExecutor = None) -> dict:
    if agent is None:
        agent = create_sql_agent()
    result = await agent.ainvoke({"input": question})
    return {"source": "sql", "result": result["output"]}
