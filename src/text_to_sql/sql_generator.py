import os
from click import prompt
from google import genai
from config import GEMINI_API_KEY


class SQLGenerator:

    def __init__(self):

        self.client = genai.Client(api_key=GEMINI_API_KEY)

    def generate_sql(self, prompt):

        response = self.client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )

        print("===== RAW RESPONSE =====")
        print(response.text)

        sql = response.text.strip()

        # IMPORTANT: remove longer code fence first
        sql = sql.replace("```sqlite", "")
        sql = sql.replace("```sql", "")
        sql = sql.replace("```", "")
        sql = sql.strip()

        return sql

from retriever import GlossaryRetriever
from prompt_builder import PromptBuilder
from config import GLOSSARY_PATH, DB_PATH, SCHEMA_DOC_PATH

if __name__ == "__main__":
    question = "What are the top 5 product categories by revenue in 202101?"

    # 1. Retrieve context from RAG
    retriever = GlossaryRetriever(GLOSSARY_PATH, DB_PATH, SCHEMA_DOC_PATH)
    retrieval_result = retriever.retrieve(question)

    # 2. Build prompt with context
    builder = PromptBuilder()
    prompt = builder.build_prompt(question, retrieval_result)

    # 3. Generate SQL using enriched prompt
    generator = SQLGenerator()
    sql = generator.generate_sql(prompt)  # ✅ pass prompt not question

    print(sql)