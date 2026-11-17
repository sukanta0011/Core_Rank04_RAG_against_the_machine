"""Qwen3-0.6B wrapper used for answer generation."""
from typing import Any, Dict, List

import torch
from pydantic import BaseModel, Field, PrivateAttr, field_validator
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.answer_generation.models.abstract_model import Model

ALLOWED_DEVICE_TYPES = ('cuda', 'cpu', None)


class SmallLLM(BaseModel, Model):
    """Qwen3-0.6B in non-thinking mode, with deterministic decoding.

    Attributes:
        model_name: Hugging Face model id.
        device_type: "cuda", "cpu", or None to pick automatically.
    """

    model_name: str = "Qwen/Qwen3-0.6B"
    device_type: str | None = None

    _tokenizer: Any = PrivateAttr()
    _model: Any = PrivateAttr()

    @field_validator("device_type", mode="before")
    @classmethod
    def validate_device_type(cls, device: str | None) -> str:
        """Check the device name and resolve None to cuda or cpu.

        Args:
            device: "cuda", "cpu" or None.

        Returns:
            The device to use. Falls back to cpu if cuda is requested
            but not available.

        Raises:
            ValueError: If the device name is unknown.
        """
        if device not in ALLOWED_DEVICE_TYPES:
            raise ValueError(
                f"'{device}' is unknown device type. "
                f"Allowed types are {ALLOWED_DEVICE_TYPES[:-1]}")
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        elif device == "cuda" and not torch.cuda.is_available():
            print("Warning: cuda requested but not available, using cpu")
            device = "cpu"
        return device

    def model_post_init(self, __context: Any) -> None:
        """Load the tokenizer and the model (slow: done once).

        float32 is used on purpose: on CPUs without native float16
        support it was about 10 times faster in our measurements.
        """
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            dtype=torch.float32,
            device_map="auto" if self.device_type == "cuda" else None,
            tie_word_embeddings=False,  # silences the tied-weights warning
        )
        self._model.eval()

    def generate_answer(
            self, resources: List[Dict],
            tokens_limit: int = 500
            ) -> str:
        """Generate an answer for a chat prompt.

        Args:
            resources: Chat messages (system and user) built by the
                prompt generator.
            tokens_limit: Maximum number of new tokens.

        Returns:
            The answer text, without the prompt and without thinking.

        Raises:
            ValueError: If `tokens_limit` is not positive.
        """
        if tokens_limit <= 0:
            raise ValueError("tokens_limit must be positive")
        prompt = self._tokenizer.apply_chat_template(
            resources, tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )
        model_inputs = self._tokenizer(
            [prompt], return_tensors="pt").to(self._model.device)

        with torch.inference_mode():
            outputs = self._model.generate(
                **model_inputs,
                max_new_tokens=tokens_limit,
                do_sample=False,  # repeatable, fewer invented details
            )

        output_ids = outputs[0][model_inputs.input_ids.shape[1]:]
        return str(self._tokenizer.decode(
            output_ids, skip_special_tokens=True)).strip()
    