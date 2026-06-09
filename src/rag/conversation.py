"""
conversation.py
───────────────
Ties memory, clarification, and the RAG chain into one conversation loop.
"""

from .memory        import create_session
from .clarification import is_vague
from .retrieval     import create_retriever
from .chain         import build_rag_chain


def ask(
    question:    str,
    vectorstore,
    child_docs:  list,
    memory,
    store,
    session_id:  str,
    year:        int  = None,
    doc_type:    str  = None,
) -> str:
    """
    Run one turn of the conversation:
      1. Check if question is vague — return clarification request if so.
      2. Inject conversation history into the RAG chain.
      3. Retrieve + generate answer.
      4. Save turn to in-memory window and SQLite.

    Returns the assistant's answer as a string.
    """
    chat_history = memory.format_history()

    # Step 1: Clarification check
    vague, clarification_q = is_vague(question, chat_history)
    if vague:
        response = f"[Clarification needed] {clarification_q}"
        memory.add_turn(question, response)
        store.save_turn(session_id, question, response)
        return response

    # Step 2: Build retriever + chain with current history
    retriever = create_retriever(vectorstore, child_docs,
                                 year=year, doc_type=doc_type)
    chain     = build_rag_chain(retriever, chat_history=chat_history)

    # Step 3: Generate answer
    answer = chain.invoke(question)

    # Step 4: Persist turn to window memory and SQLite
    memory.add_turn(question, answer)
    store.save_turn(session_id, question, answer)

    return answer