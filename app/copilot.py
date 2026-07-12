"""
app/copilot.py
──────────────
Unified entry point for the AI Analytics Copilot.

One conversation loop that routes each question to internal Text-to-SQL,
external RAG, or both — then merges the results into a PM report. This is the
single end-to-end demo (the SQL-only loop in app/app.py still exists for the
narrow Text-to-SQL demo).

Run from the project root:
    python -m app.copilot
"""

import sys
import asyncio
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))
load_dotenv()  # HF_TOKEN

from src.router import RouterAgent, PMReport
from src.sql_agent.config import GLOSSARY_PATH, DB_PATH


def render(report: PMReport) -> None:
    print("\n===== PM REPORT =====")
    print(f"Summary: {report.summary}")

    if report.comparison_table:
        print("\nComparison:")
        for row in report.comparison_table:
            print(
                f"  - {row.get('metric', '?')}: "
                f"you={row.get('your_value', 'N/A')} | "
                f"industry={row.get('industry_avg', 'N/A')} | "
                f"{row.get('status', '')}"
            )

    if report.action_items:
        print("\nAction items:")
        for i, item in enumerate(report.action_items, 1):
            print(f"  {i}. {item}")

    if report.data_sources:
        print(f"\nSources: {', '.join(report.data_sources)}")
    print()


def main():
    print("Loading copilot (SQL glossary index + report vector store)...")
    agent = RouterAgent(glossary_path=GLOSSARY_PATH, db_path=DB_PATH, session_id="cli")

    print("\nAI Analytics Copilot ready. Ask about internal KPIs, industry")
    print("reports, or how the two compare. Type 'exit' to quit.\n")

    while True:
        question = input("What can I help you with today?\n> ").strip()
        if question.lower() in ("exit", "quit"):
            print("Bye! Have a nice day.")
            break
        if not question:
            continue

        report = asyncio.run(agent.run(question))
        render(report)


if __name__ == "__main__":
    main()
