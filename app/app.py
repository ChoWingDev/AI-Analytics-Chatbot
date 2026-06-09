import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.sql_agent.sql_pipeline import TextToSQLPipeline
from src.sql_agent.config import GLOSSARY_PATH, DB_PATH

def main():
    sql_pipeline = TextToSQLPipeline(
        glossary_path=GLOSSARY_PATH,
        db_path=DB_PATH
    )
    print("AI Analytics Copilot Started")
    print("Type 'exit' to quit.\n")

    while True:
        question = input("What can I help you today?\n")
        
        if question.lower() in ["exit", "quit"]:
            print("Bye! Have a nice day")
            break
        
        result = sql_pipeline.run(question)

        print("\n===== ANSWER =====")
        print(result["result"])
        print()

if __name__=="__main__":
    main()