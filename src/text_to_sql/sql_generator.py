import os
import google.generativeai as genai
from config import GEMINI_API_KEY


class SQLGenerator:

    def __init__(self):

        genai.configure(
            api_key=GEMINI_API_KEY
        )

        self.model = genai.GenerativeModel("models/gemini-2.0-flash-lite")

    def generate_sql(self, prompt):

        response = self.model.generate_content(prompt)

        print("===== RAW RESPONSE =====")
        print(response.text)

        sql = response.text.strip()

        # IMPORTANT: remove longer code fence first
        sql = sql.replace("```sqlite", "")
        sql = sql.replace("```sql", "")
        sql = sql.replace("```", "")
        sql = sql.strip()

        return sql

if __name__ == "__main__":
    """
    for model in genai.list_models():
            if "generateContent" in model.supported_generation_methods:
                print(model.name)
    """

    prompt = """
    You are a SQL expert.
    Use SQLite syntax only.

    Question:
    What was the total revenue last month?

    Generate SQL only.
    """
    print("===== PROMPT =====")
    print(prompt)

    generator = SQLGenerator()

    sql = generator.generate_sql(prompt)

    print(sql)