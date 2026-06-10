from sql_graph import create_sql_graph
from config import GLOSSARY_PATH, DB_PATH

graph = create_sql_graph(
    glossary_path = GLOSSARY_PATH,
    db_path = DB_PATH
)

result = graph.invoke({
    "question": "What is total revenue?"
})

print(result)