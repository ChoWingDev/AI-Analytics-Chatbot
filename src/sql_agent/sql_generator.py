from src.llm import get_llm


class SQLGenerator:
    """Generates SQLite queries from an enriched prompt using the shared HF LLM."""

    def __init__(self):
        # Tier-1: standardized on HuggingFace (was Gemini). Deterministic-ish,
        # enough headroom for multi-line SQL with CTEs.
        self.llm = get_llm(max_new_tokens=512, temperature=0.0)

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
