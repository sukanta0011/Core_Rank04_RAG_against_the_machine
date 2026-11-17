"""Prompt construction for grounded answers."""
from typing import Dict, List

NO_ANSWER_TEXT = "I cannot find this in the provided context."
MAX_CHARS_PER_CHUNK = 1500  # keeps the prompt short on a CPU

SYSTEM_PROMPT = (
    "You are a vLLM expert. Answer the question using ONLY the numbered "
    "context passages. Do not add facts, names, numbers or code that are "
    "not in the passages. If the passages do not contain the answer, "
    f"reply exactly: {NO_ANSWER_TEXT} "
    "Give a short, direct answer with no introduction."
)


def format_context(context: str | List[str]) -> str:
    """Turn the retrieved chunks into numbered blocks.

    A Python list printed with f-string gives quotes and escaped
    newlines (`['a\\nb']`), which is noisy for a small model.

    Args:
        context: One text, or the list of chunk texts (best first).

    Returns:
        Blocks like `[1]\\n<text>`, separated by blank lines.
    """
    chunks = [context] if isinstance(context, str) else list(context)
    blocks = []
    for number, text in enumerate(chunks, start=1):
        text = str(text).strip()
        if len(text) > MAX_CHARS_PER_CHUNK:
            text = text[:MAX_CHARS_PER_CHUNK] + " ..."
        blocks.append(f"[{number}]\n{text}")
    return "\n\n".join(blocks)


class InitialPromptGenerator:
    """Builds the chat messages sent to the language model."""

    @staticmethod
    def get_type1_prompt(
        question: str, context: str | List[str]
            ) -> List[Dict]:
        """Build a grounded question-answering prompt.

        The question comes after the context and the rule is repeated
        right before it, because small models weigh the end of the
        prompt most.

        Args:
            question: The user's question.
            context: Retrieved chunk text(s), best first.

        Returns:
            Chat messages: a system rule and one user message.
        """
        user = (
            f"Context:\n{format_context(context)}\n\n"
            f"Question: {question}\n"
            "Answer using only the context above. If it is not there, "
            f"say: {NO_ANSWER_TEXT}"
        )
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ]
