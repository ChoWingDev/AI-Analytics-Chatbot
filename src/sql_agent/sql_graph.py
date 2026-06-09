from typing import TypedDict, Any
from langgraph.graph import StateGraph, END

from .retriever import GlossaryRetriever
from .prompt_builder import PromptBuilder
from .sql_executor import SQLExecutor


class SQLAgentState(TypedDict):
    """
    Shared state passed between LangGraph nodes.

    question
        Original user question

    retrieval_result
        Retrieved glossary metrics, business rules,
        time filters, and schema context

    prompt
        Final prompt sent to the SQL generation model

    sql
        Generated SQL statement

    result
        SQL execution result returned from SQLite
    """
    question: str
    retrieval_result: dict
    prompt: str
    sql: str
    result: Any


def create_sql_graph(glossary_path, db_path):
    """
    Build Text-to-SQL LangGraph workflow.

    Flow:

    User Question
            ↓
    Retrieve Context
            ↓
    Build Prompt
            ↓
    Generate SQL
            ↓
    Execute SQL
            ↓
        Result

    Returns:
        Compiled LangGraph application
    """

    # Initialize reusable components
    retriever = GlossaryRetriever(glossary_path, db_path)
    prompt_builder = PromptBuilder()
    executor = SQLExecutor(db_path)

    def retrieve_context(state: SQLAgentState):
        """
        Retrieve relevant business knowledge
        from glossary and database schema.

        Output:
            retrieval_result
        """
        retrieval_result = retriever.retrieve(
            state["question"]
        )

        return {
            "retrieval_result": retrieval_result
        }

    def build_prompt(state: SQLAgentState):
        """
        Construct Text-to-SQL prompt using:

        - user question
        - metric definitions
        - business rules
        - schema context
        - time period logic

        Output:
            prompt
        """
        prompt = prompt_builder.build_prompt(
            state["question"],
            state["retrieval_result"]
        )

        return {
            "prompt": prompt
        }

    def mock_generate_sql(state: SQLAgentState):
        """
        Temporary SQL generator.

        Used during LangGraph development
        to avoid consuming Gemini quota.

        Later this node will be replaced by:

            SQLGenerator.generate_sql()

        Output:
            sql
        """
        return {
            "sql": """
            SELECT
                SUM(order_revenue) AS revenue
            FROM mart_order_summary
            WHERE status = 'Complete'
            """
        }

    def execute_sql(state: SQLAgentState):
        """
        Execute generated SQL against SQLite.

        Output:
            result (Pandas DataFrame)
        """
        result = executor.execute(
            state["sql"]
        )

        return {
            "result": result
        }

    # Create graph definition
    graph = StateGraph(SQLAgentState)

    # Register graph nodes
    graph.add_node(
        "retrieve_context",
        retrieve_context
    )

    graph.add_node(
        "build_prompt",
        build_prompt
    )

    graph.add_node(
        "mock_generate_sql",
        mock_generate_sql
    )

    graph.add_node(
        "execute_sql",
        execute_sql
    )

    # Define execution order
    graph.set_entry_point(
        "retrieve_context"
    )

    graph.add_edge(
        "retrieve_context",
        "build_prompt"
    )

    graph.add_edge(
        "build_prompt",
        "mock_generate_sql"
    )

    graph.add_edge(
        "mock_generate_sql",
        "execute_sql"
    )

    graph.add_edge(
        "execute_sql",
        END
    )

    return graph.compile()