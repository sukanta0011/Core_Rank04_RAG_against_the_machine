from typing import List, Dict

class InitialPromptGenerator:
    @staticmethod
    def get_type1_prompt(question: str, context: str | List[str]) -> List[Dict]:
        messages = [
            {"role": "system", "content": "You are a vLLM expert. Use the context to answer."},
        ]
        if isinstance(context, str) or isinstance(context, list):
            messages.append(
                {"role": "user", "content": f"Context: {context}\n\nQuestion: {question}"}
            )
        return messages