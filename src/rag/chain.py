"""
chain.py
────────
RAG chain assembly: prompt + LLM + output parser.
"""

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

from .llm import get_llm


def format_docs(docs: list[Document]) -> str:
    """
    Format retrieved chunks into a context block for the LLM prompt.
    Includes source, page, year, and doc_type for accurate citations.
    """
    formatted = []
    for doc in docs:
        header = (
            f"[Source: {doc.metadata.get('source', 'unknown')} | "
            f"Page: {doc.metadata.get('page', '?')} | "
            f"Year: {doc.metadata.get('year', 'unknown')} | "
            f"Type: {doc.metadata.get('doc_type', 'unknown')}]"
        )
        formatted.append(f"{header}\n{doc.page_content}")
    return "\n\n---\n\n".join(formatted)


def build_rag_chain(retriever, chat_history: str = ""):
    """
    Build the RAG chain: retriever → format → prompt → LLM → string.

    Prompt design:
    - Chat history section only appears when history exists (avoids
      confusing the LLM with an empty section).
    - Citations required: source, page, year, doc_type in every answer.
    - Hallucination guard: say "I don't have enough data" not guess.
    - Distinction: separate company data from industry benchmarks.
    """
    history_section = ""
    if chat_history:
        history_section = f"""
        Previous conversation (use this to resolve follow-up questions):
        {chat_history}
        """

    prompt = ChatPromptTemplate.from_template(f"""
    You are an ecommerce industry analyst assistant helping a Product Manager.
    Answer the question using ONLY the context provided below.
    If the context does not contain enough information, say:
    "I don't have enough data in the loaded reports to answer this confidently."
    {history_section}
    When citing data always include: source document, page number, year, and
    whether it is a company report or an industry benchmark.
    Clearly distinguish company-specific data from industry-wide benchmarks.

    Context:
    {{context}}

    Question: {{question}}

    Answer:""")

    return (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | get_llm()
        | StrOutputParser()
    )