import re

from src.llm import get_llm

# A fenced block, with or without a language tag: ```sql ... ```
_FENCE = re.compile(r"```(?:sqlite|sql)?\s*(.*?)```", re.DOTALL)
# Where a statement actually begins.
_SQL_START = re.compile(r"\b(WITH|SELECT)\b", re.IGNORECASE)


def extract_sql(response: str) -> str:
    """
    Pull the SQL statement out of an LLM response.

    The prompt asks for SQL only, but models routinely wrap the query in
    explanation anyway. Stripping fence markers alone left that prose in
    place and handed it to SQLite verbatim, which fails with a DatabaseError
    whose message is the model's paragraph.

    Search order — first candidate containing a statement wins:
      1. each fenced code block, in order
      2. the whole response

    Within a candidate, keep from the first SELECT/WITH up to and including
    the first semicolon, which drops any commentary that follows the query.
    A response with nothing recognisable is returned unchanged so the
    executor surfaces a real error rather than silently running something else.
    """
    text = response.strip()

    for candidate in [m.group(1) for m in _FENCE.finditer(text)] + [text]:
        match = _SQL_START.search(candidate)
        if not match:
            continue
        sql = candidate[match.start():]
        if ";" in sql:
            sql = sql[: sql.index(";") + 1]
        return sql.strip()

    return text


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

        return extract_sql(response)


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
