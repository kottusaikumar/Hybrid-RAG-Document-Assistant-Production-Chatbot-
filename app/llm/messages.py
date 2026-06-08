"""
app/llm/messages.py
-------------------
Builds the LangChain message list passed to the LLM.
Kept separate from the client so it can be tested independently.
"""

from langchain.schema import HumanMessage, SystemMessage

from app.config import settings


def build_messages(
    query: str,
    context: str,
    history: list,
    previous_answer: str,
    intent_instruction: str,
) -> list:
    """
    Assemble the full message list:
      [SystemMessage] + [history window] + [HumanMessage with context]

    If previous_answer is non-empty, includes it plus the intent instruction
    so the model knows exactly what kind of follow-up response is expected.
    """
    messages = [SystemMessage(content=settings.SYSTEM_PROMPT)]
    messages.extend(history[-settings.HISTORY_WINDOW:])

    if previous_answer:
        user_content = (
            f"Context from PDF:\n\n{context}\n\n---\n\n"
            f"Previous answer already given to the user:\n{previous_answer[:200]}\n\n---\n\n"
            f"User follow-up question: {query}\n\n"
            f"Instruction: {intent_instruction}"
        )
    else:
        user_content = f"Context from PDF:\n\n{context}\n\n---\n\nQuestion: {query}"

    messages.append(HumanMessage(content=user_content))
    return messages
