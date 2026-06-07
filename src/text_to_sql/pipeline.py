from retriever import GlossaryRetriever
from prompt_builder import PromptBuilder
from sql_generator import SQLGenerator
from sql_executor import SQLExecutor
from sql_evaluator import evaluate_sql

class TextToSQLPipeline:
    def __init__(self, glossary_path, db_path):
        self.retriever = GlossaryRetriever(glossary_path, db_path)
        self.prompt_builder = PromptBuilder()
        self.sql_generator = SQLGenerator()
        self.sql_executor = SQLExecutor(db_path)

    def run(self, question):

        retrieval_result = self.retriever.retrieve(question)
        prompt = self.prompt_builder.build_prompt(
            question,
            retrieval_result
        )
        print(prompt)

        sql = self.sql_generator.generate_sql(prompt)

        result = self.sql_executor.execute(sql)

        return {
            "question": question,
            "retrieval_result": retrieval_result,
            "prompt": prompt,
            "sql": sql,
            "result": result
        }
    
if __name__ == "__main__":
    pipeline = TextToSQLPipeline(
        glossary_path = "/Users/chowingchan/Desktop/Project/AI-Analytics-Copilot/AI-Analytics-Copilot/config/glossary.json",
        db_path="/Users/chowingchan/Desktop/Project/AI-Analytics-Copilot/Competitive-Intelligence-Internal-Analytics-System/data/database/thelook_ecommerce.db"
    )

    output = pipeline.run("What is total revenue in last 30 days?")
    
    print("\n===== GENERATED SQL =====")
    print(output["sql"])

    print("\n===== RESULT =====")
    print(output["result"])