from src.llm import get_sql_llm


class SQLGenerator:
    """Generates SQLite queries from an enriched prompt using the code-tuned HF model."""

    def __init__(self):
        # Same provider (HuggingFace), but a SQL-specialist model. The general
        # chat model kept inventing columns and answering with several statements;
        # swapping in a code-tuned model is the cheapest lever on SQL accuracy.
        # temperature=0.0 — we want the same question to yield the same query.
        self.llm = get_sql_llm(max_new_tokens=512, temperature=0.0)

    def generate_sql(self, prompt):
        response = self.llm.invoke(prompt)

        print("===== RAW RESPONSE =====")
        print(response)

        sql = response.strip()

        # IMPORTANT: remove longer code fence first
        sql = sql.replace("```sqlite", "")
        sql = sql.replace("```sql", "")
        sql = sql.replace("```", "")
        sql = sql.strip()

        return sql


from .retriever import GlossaryRetriever
from .prompt_builder import PromptBuilder
from .config import GLOSSARY_PATH, DB_PATH

if __name__ == "__main__":
    question = "What are the top 5 product categories by revenue in 202101?"

    # 1. Retrieve context from the glossary + live schema
    retriever = GlossaryRetriever(GLOSSARY_PATH, DB_PATH)
    retrieval_result = retriever.retrieve(question)

    # 2. Build prompt with context
    builder = PromptBuilder()
    prompt = builder.build_prompt(question, retrieval_result)

    # 3. Generate SQL using the enriched prompt (not the raw question)
    generator = SQLGenerator()
    sql = generator.generate_sql(prompt)

    print(sql)
