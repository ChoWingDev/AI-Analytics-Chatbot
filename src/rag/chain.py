"""
chain.py
────────
RAG chain assembly: prompt + LLM + output parser.
Receives a retriever, returns a runnable chain.
"""

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from .llm import get_llm


def format_docs(docs: list[Document]) -> str:
    """
    Format retrieved chunks into a context block for the prompt.
    Includes source, page, year, and doc_type so the LLM can cite accurately
    and distinguish company data from industry benchmarks.
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


def build_rag_chain(retriever):
    """
    Build the RAG chain: retriever → format → prompt → LLM → string.

    Prompt design choices:
    - Citations required: source, page, year, doc_type in every answer.
    - Hallucination guard: model must say "I don't have enough data" rather
      than inventing statistics when context is insufficient.
    - Distinction instruction: model separates company-specific data from
      industry-wide benchmarks (important since both are in the corpus).
    """
    prompt = ChatPromptTemplate.from_template("""
You are an ecommerce industry analyst assistant helping a Product Manager.
Answer the question using ONLY the context provided below.
If the context does not contain enough information, say:
"I don't have enough data in the loaded reports to answer this confidently."

When citing data always include: source document, page number, year, and
whether it is a company report or an industry benchmark.
Clearly distinguish company-specific data from industry-wide benchmarks.

Context:
{context}

Question: {question}

Answer:""")

    return (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | get_llm()
        | StrOutputParser()
    )
