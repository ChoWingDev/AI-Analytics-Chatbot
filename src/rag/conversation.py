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


def run_scripted_conversation(vectorstore, child_docs, session_id: str = "conversation demo"):
    """
    Demo: 5 consecutive follow-up questions.

    Designed to test multi-turn context:
      T1 — Grounding question (establishes the topic in memory)
      T2 — Follow-up using "that" (tests pronoun resolution via history)
      T3 — Causal follow-up (tests retention of prior answer)
      T4 — Cross-company pivot (tests context switch with history intact)
      T5 — Vague question (tests clarification layer)

    Without memory, T2 and T3 would fail because the LLM would not know
    what "that" or "that growth" refer to. With the window memory injected
    into the prompt, it resolves correctly.
    """
    print("\n" + "=" * 60)
    print("Demo - Multi-Turn Conversation")
    print(f"Session ID: {session_id}")
    print("=" * 60)

    memory, store, session_id = create_session(session_id, k=5)

    turns = [
        {
            "question": "What was Aritzia eCommerce net revenue in fiscal 2025 and how did it grow?",
            "year":     2025,
            "doc_type": "company_report",
            "note":     "T1 — Grounding: establishes $951M revenue figure in memory",
        },
        {
            "question": "What was the eCommerce 2.0 platform Aritzia launched to support that growth?",
            "year":     2025,
            "doc_type": "company_report",
            "note":     "T2 — Follow-up: 'that growth' resolves to T1; eCommerce 2.0 is explicitly in the doc",
        },
        {
            "question": "What digital marketing investments did Aritzia make alongside that platform launch?",
            "year":     2025,
            "doc_type": "company_report",
            "note":     "T3 — Causal: 'that platform launch' resolves from T2; digital marketing is in the doc",
        },
        {
            "question": "According to the industry logistics report, what percentage of carts are abandoned and why?",
            "year":     2024,
            "doc_type": "industry_benchmark",
            "note":     "T4 — Pivot to benchmark data; tests context switch while retaining session history",
        },
        {
            "question": "What about risks?",
            "year":     None,
            "doc_type": None,
            "note":     "T5 — Vague: should trigger clarification layer",
        },
    ]

    for i, turn in enumerate(turns, 1):
        print(f"\n{'─' * 60}")
        print(f"Turn {i} — {turn['note']}")
        print(f"Q: {turn['question']}")
        print(f"Memory: {memory.turn_count} turn(s) in window")
        print("─" * 40)

        answer = ask(
            question=turn["question"],
            vectorstore=vectorstore,
            child_docs=child_docs,
            memory=memory,
            store=store,
            session_id=session_id,
            year=turn["year"],
            doc_type=turn["doc_type"],
        )

        print(f"A: {answer}")

    print(f"\n{'=' * 60}")
    print(f"Session complete — {memory.turn_count} turns in memory window.")
    print(f"Full history saved to sessions.db (session_id='{session_id}')")
    print("=" * 60)