from typing import Any, List, Dict
import torch
from pydantic import (
    BaseModel, PrivateAttr,
    field_validator, ValidationError, Field)
from transformers import AutoModelForCausalLM, AutoTokenizer
# from vllm import LLM, SamplingParams
from src.answer_generation.models.abstract_model import Model


def hugging_face_example():
    model_name = "Qwen/Qwen3-0.6B"

    # load the tokenizer and the model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )

    # prepare the model input
    prompt = "Give me a short introduction to large language model."
    messages = [
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        # Switches between thinking and non-thinking modes.
        enable_thinking=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # conduct text completion
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=32768
    )
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

    # # parsing thinking content
    # try:
    #     # rindex finding 151668 (</think>)
    #     index = len(output_ids) - output_ids[::-1].index(151668)
    # except ValueError:
    index = 0

    # thinking_content = tokenizer.decode(
    #     output_ids[:index], skip_special_tokens=True).strip("\n")
    content = tokenizer.decode(
        output_ids[index:], skip_special_tokens=True).strip("\n")

    # print("thinking content:", thinking_content)
    print("content:", content)


ALLOWED_DEVICE_TYPES = ('cuda', 'cpu', None)

# class SmallvLLM(BaseModel, Model):
#     _llm: LLM = PrivateAttr()

#     def model_post_init(self, __context: Any) -> None:
#         self._llm = LLM(
#             model="Qwen/Qwen3-0.6B",
#             gpu_memory_utilization=0.7,
#             trust_remote_code=True,
#             max_model_len=2048
#         )

#     def generate_answer(
#             self, resources: List[Dict],
#             tokens_limit: int = Field(gt=0, default=500)
#             ) -> str:
#         sampling_params = SamplingParams(temperature=0.3, max_tokens=256)
#         answers = self._llm.generate([resources], sampling_params)
#         return answers[0].outputs[0].text


class SmallLLM(BaseModel, Model):
    model_name: str = "Qwen/Qwen3-0.6B"
    device_type: str | None = None

    _tokenizer: Any = PrivateAttr()
    _model: Any = PrivateAttr()

    @field_validator("device_type", mode="before")
    @classmethod
    def validate_device_type(cls, device: str | None) -> str:
        if device not in ALLOWED_DEVICE_TYPES:
            raise ValidationError(
                f"'{device}' is unknown device type. "
                f"Allowed types are {ALLOWED_DEVICE_TYPES[:-1]}"
            )
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        return device

    def model_post_init(self, __context: Any) -> None:
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map="auto" if self.device_type == "cuda" else None,
            trust_remote_code=True
        )

    def generate_answer(
            self, resources: List[Dict],
            tokens_limit: int = Field(gt=0, default=500)
            ) -> str:
        prompt = self._tokenizer.apply_chat_template(
            resources, tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )
        model_inputs = self._tokenizer(
            [prompt], return_tensors="pt").to(self.device_type)

        outputs = self._model.generate(
            **model_inputs,
            max_new_tokens=tokens_limit
        )

        output_ids = outputs[0][len(model_inputs.input_ids[0]):].tolist()
        content = self._tokenizer.decode(
            output_ids, skip_special_tokens=True).strip("\n")
        return content


# def test_small_vllm() -> None:
#     llm = SmallvLLM()
#     prompt = "This a test prompt"

if __name__ == "__main__":
    hugging_face_example()
