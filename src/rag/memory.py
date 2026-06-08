"""
memory.py
─────────
Conversation memory with two layers:

  1. In-memory window  — keeps the last k turns in RAM for fast access
                         during a session. 

  2. SQLite persistence — writes every turn to disk so history survives
                          restarts and can be replayed across sessions.
"""

import sqlite3
from dataclasses import dataclass
from datetime import datetime


# ── Data structure ────────────────────────────────────────────────────────────

@dataclass
class Turn:
    """One round of conversation: a human question and the assistant's answer."""
    human: str
    ai:    str


# ── In-memory window memory ───────────────────────────────────────────────────

class ConversationMemory:
    """
    Sliding window over the last k conversation turns.

    How it's used:
    Before each RAG call, format_history() produces a plain-text block
    of recent turns that gets injected into the prompt as {chat_history}.
    The LLM can then resolve pronouns and follow-ups correctly —
    e.g. "How does that compare?" → the LLM knows "that" = last answer.
    """

    def __init__(self, k: int = 5):
        self.k: int             = k
        self._turns: list[Turn] = []

    def add_turn(self, human: str, ai: str) -> None:
        """Add one turn. If buffer is full, drop the oldest turn (FIFO)."""
        self._turns.append(Turn(human=human, ai=ai))
        if len(self._turns) > self.k:
            self._turns.pop(0)

    def format_history(self) -> str:
        """
        Render recent turns as plain text for prompt injection.

        Example output:
          Human: What was Aritzia's revenue?
          Assistant: Aritzia's eCommerce net revenue was $951M in 2025...

          Human: How does that compare to 2024?
          Assistant: In 2024 it was $774M, so growth was 23%...
        """
        if not self._turns:
            return ""
        lines = []
        for turn in self._turns:
            lines.append(f"Human: {turn.human}")
            lines.append(f"Assistant: {turn.ai}")
        return "\n\n".join(lines)

    def clear(self) -> None:
        """Reset memory — call this to start a new session."""
        self._turns = []

    @property
    def turn_count(self) -> int:
        return len(self._turns)

