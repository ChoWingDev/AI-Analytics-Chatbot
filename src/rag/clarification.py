"""
clarification.py
────────────────
Detects vague questions and asks for clarification before running RAG.

Why this matters:
A question like "tell me more" or "what about risks?" has no grounding —
the retriever has nothing specific to search for and will return poor results.
Instead of guessing and producing a bad answer, the pipeline asks the user
to be more specific. 

How it works:
The question is sent to the LLM with a simple classification prompt.
The LLM returns "CLEAR" or "VAGUE: <clarification question>".
If vague, the conversation runner shows the clarification question instead
of running the RAG chain.
"""

from .llm import get_llm

# Phrases that are almost always vague follow-ups without context.
ALWAYS_VAGUE = [
    "tell me more",
    "what about",
    "and then",
    "can you explain",
    "go on",
    "more details",
    "elaborate",
    "what else",
    "how so",
    "why is that",
]


def is_vague(question: str, chat_history: str = "") -> tuple[bool, str]:
    """
    Check if a question is too vague to retrieve meaningful results.

    Two-stage check:
    1. Fast heuristic — matches known vague patterns or very short questions
       with no history. No LLM call needed, returns immediately.
    2. LLM classification — for borderline cases, asks the LLM to decide.
       Skipped entirely if the question is clearly long and specific.

    Parameters:
      question     — the user's raw question
      chat_history — recent turns from ConversationMemory.format_history().
                     A short question with history may still be clear:
                     "How does that compare?" after a revenue answer is fine.

    Returns:
      (False, "")                    — clear, proceed with RAG
      (True, "Could you clarify X?") — vague, show this to the user instead
    """
    q          = question.strip().lower()
    word_count = len(q.split())

    # Stage 1: fast heuristics (no LLM cost)
    if word_count <= 3 and not chat_history:
        return True, "Could you be more specific? What would you like to know about?"

    if any(q.startswith(phrase) for phrase in ALWAYS_VAGUE) and not chat_history:
        return True, (
            "Could you be more specific? For example: "
            "which company, which metric, or which time period are you asking about?"
        )

    # Stage 2: LLM check for borderline cases only
    # Skip if clearly specific (long questions rarely need clarification)
    if word_count > 10:
        return False, ""

    llm = get_llm()

    context_block = f"\nRecent conversation:\n{chat_history}\n" if chat_history else ""

    prompt = f"""You are helping decide if a question needs clarification before searching a document database.{context_block}
            Question: "{question}"

            Is this question specific enough to search for, or is it too vague?
            - If the recent conversation provides enough context to understand what is being asked, it is CLEAR.
            - If it is impossible to know what documents to retrieve, it is VAGUE.

            Reply with exactly one of:
            CLEAR
            VAGUE: <one short clarifying question to ask the user>"""

    result = llm.invoke(prompt).strip()

    if result.upper().startswith("CLEAR"):
        return False, ""
    elif result.upper().startswith("VAGUE"):
        clarification = result.split(":", 1)[-1].strip() if ":" in result else (
            "Could you be more specific about what you are looking for?"
        )
        return True, clarification
    else:
        return False, ""  # default to clear on unexpected output