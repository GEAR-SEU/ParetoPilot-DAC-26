import ast
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from config_loader import get_config_value


DEFAULT_SYSTEM_PROMPT = (
    "You are an expert in high-level synthesis design space exploration. "
    "Use the provided HLS code and pragma configuration parameters as context "
    "for generating configurations. Avoid duplicates and ensure each tuple is unique."
)


@dataclass
class APIChatModelFactory:
    api_key: str | None = None
    base_url: str | None = None
    model_name: str = "gpt-4o-mini"

    @classmethod
    def from_config(cls) -> "APIChatModelFactory":
        api_key = get_config_value("llm", "api_key", default="") or None
        base_url = get_config_value("llm", "api_base", default="") or None
        model_name = get_config_value("llm", "model", default="")
        return cls(api_key=api_key, base_url=base_url, model_name=model_name)

    def create(self, temperature: float) -> Any:
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            model=self.model_name,
            temperature=temperature,
        )


@dataclass
class HFChatModelFactory:
    model_name: str = "XXXiong/ParetoPilot"

    @classmethod
    def from_config(cls) -> "HFChatModelFactory":
        return cls(
            model_name=get_config_value("llm", "hf_model", default="XXXiong/ParetoPilot"),
        )


class SamplerBackend(Protocol):
    def invoke(self, system_prompt: str, prompt: str) -> dict[str, Any]:
        ...


@dataclass
class APIBackend:
    chat: Any

    def invoke(self, system_prompt: str, prompt: str) -> dict[str, Any]:
        from langchain.schema import HumanMessage, SystemMessage

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=prompt),
        ]
        response = self.chat.invoke(messages)
        return {
            "content": response.content,
            "token_usage": response.response_metadata.get("token_usage", {}),
            "response_metadata": response.response_metadata,
        }


@dataclass
class HFBackend:
    model_name: str
    _tokenizer: Any = None
    _model: Any = None

    def _load_model(self) -> None:
        if self._tokenizer is not None and self._model is not None:
            return

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
        except ImportError as exc:
            raise ImportError(
                "transformers and torch are required for Hugging Face local inference. "
                "Please install the dependencies in requirements.txt."
            ) from exc

        print(f"Loading Hugging Face model: {self.model_name}")
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )

    def invoke(self, system_prompt: str, prompt: str) -> dict[str, Any]:
        self._load_model()
        combined_prompt = f"{system_prompt}\n\n{prompt}"
        messages = [{"role": "user", "content": combined_prompt}]
        text = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        model_inputs = self._tokenizer([text], return_tensors="pt").to(self._model.device)

        generation_output = self._model.generate(
            **model_inputs,
            max_new_tokens=8192,
            temperature=0.7,
            do_sample=False,
            top_k=50,
            top_p=0.95,
        )
        generated_ids = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(model_inputs.input_ids, generation_output)
        ]
        content = self._tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        prompt_tokens = int(model_inputs["input_ids"].shape[-1])
        completion_tokens = int(generated_ids[0].shape[-1]) if generated_ids else 0

        return {
            "content": content,
            "token_usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
            "response_metadata": {
                "backend": "hf",
                "model_name": self.model_name,
                "dtype": "bfloat16",
                "device_map": "auto",
                "max_new_tokens": 8192,
            },
        }


@dataclass
class LLMPragmaSampler:
    backend: SamplerBackend

    @classmethod
    def from_config(
        cls,
        temperature: float = 0.7,
        backend: str = "hf",
        hf_model: str | None = None,
    ) -> "LLMPragmaSampler":
        if backend == "api":
            factory = APIChatModelFactory.from_config()
            return cls(backend=APIBackend(chat=factory.create(temperature)))

        if backend == "hf":
            factory = HFChatModelFactory.from_config()
            if hf_model:
                factory.model_name = hf_model
            return cls(backend=HFBackend(model_name=factory.model_name))

        raise ValueError(f"Unsupported llm backend: {backend}")

    @staticmethod
    def build_prompt(kernel_code: str, pragma_config: str, num_sample: int) -> str:
        return f"""You are tasked with generating {num_sample} unique and effective pragma configuration combinations for optimizing the provided High-Level Synthesis design.
These configurations must target the following optimization goals while strictly adhering to the condition that only parameters available in the provided configuration file can be used.
If a parameter is not available, set the corresponding value to 0.

- Performance optimization: achieve the best performance, which will result in higher hardware consumption.
- Resource utilization optimization: minimize hardware resource usage, which may result in reduced performance.
- PPA tradeoff optimization: find a balanced unroll factor combination that offers a good tradeoff between performance and hardware utilization.

The configuration file provides parameters for pragmas:
1. `array_partition`
- `type`: `0` represents complete partitioning, `1` indicates block partitioning, and `2` indicates cyclic partitioning.
- `dim`: specifies the dimension to partition.
- `factor`: indicates the partitioning factor for block or cyclic partition types.

2. `pipeline`: specifies whether to pipeline the current loop.

3. `unroll`: specifies the unrolling factor for the current loop.

Make sure every loop and array dimension has a configuration. If it is unnecessary, set `unroll` and `factor` to 0.

Code:
```cpp
{kernel_code}
```

Available pragma parameters:
```
{pragma_config}
```

Expected output format:
```python
[
    (
        (
            {{'name': [loop_name], 'pipeline': [value], 'unroll': [value]}},
            {{...}}
        ),
        (
            {{'name': [array_name], 'type': [value], 'dim': [value], 'factor': [value]}},
            {{...}}
        )
    ),
    ...
]
```

Generate exactly {num_sample} tuples.
Output only the configurations in the required format.
"""

    @staticmethod
    def extract_list_from_response(response: str) -> list[Any]:
        match = re.search(r"```python(.*?)```", response, re.DOTALL)
        if not match:
            list_str = response.strip()
        else:
            list_str = match.group(1).strip()

        try:
            parsed_list = ast.literal_eval(list_str)
        except Exception as exc:
            raise ValueError(f"Failed to parse list from response: {exc}") from exc

        if not isinstance(parsed_list, list):
            raise ValueError("Parsed content is not a list.")

        return parsed_list

    @staticmethod
    def read_text_file(file_path: str | Path) -> str:
        path = Path(file_path)
        if not path.exists():
            print(f"File path does not exist: {path}")
            return ""

        with open(path, "r", encoding="utf-8") as file:
            return file.read()

    def sample_with_metadata(self, kernel_code_path: str | Path, pragma_config_path: str | Path, num_sample: int):
        kernel_code = self.read_text_file(kernel_code_path)
        pragma_config = self.read_text_file(pragma_config_path)
        prompt = self.build_prompt(kernel_code, pragma_config, num_sample)

        response = self.backend.invoke(DEFAULT_SYSTEM_PROMPT, prompt)
        print(response["content"])

        try:
            combinations = self.extract_list_from_response(response["content"])
        except ValueError as exc:
            print(exc)
            combinations = []

        return {
            "kernel_code_path": str(kernel_code_path),
            "pragma_config_path": str(pragma_config_path),
            "num_sample": num_sample,
            "prompt": prompt,
            "system_prompt": DEFAULT_SYSTEM_PROMPT,
            "response_content": response["content"],
            "token_usage": response.get("token_usage", {}),
            "response_metadata": response.get("response_metadata", {}),
            "combinations": combinations,
        }

    def sample(self, kernel_code_path: str | Path, pragma_config_path: str | Path, num_sample: int) -> list[Any]:
        return self.sample_with_metadata(kernel_code_path, pragma_config_path, num_sample)["combinations"]


def discretize_dataset(data):
    discretized_feature = []
    for feature in data:
        discretized_feature.append(list(np.linspace(0, 1, len(feature))))
    return np.array(discretized_feature)


def sample_prompt(kernel_code, pragma_config, num_sample):
    return LLMPragmaSampler.build_prompt(kernel_code, pragma_config, num_sample)


def extract_list_from_response(response: str):
    return LLMPragmaSampler.extract_list_from_response(response)


def llm_sampling(kernel_code_path, pragma_config_path, num_sample, backend="hf", hf_model=None):
    sampler = LLMPragmaSampler.from_config(backend=backend, hf_model=hf_model)
    return sampler.sample(kernel_code_path, pragma_config_path, num_sample)


def llm_sampling_with_metadata(kernel_code_path, pragma_config_path, num_sample, backend="hf", hf_model=None):
    sampler = LLMPragmaSampler.from_config(backend=backend, hf_model=hf_model)
    return sampler.sample_with_metadata(kernel_code_path, pragma_config_path, num_sample)


def discretize_dataset_sample(data):
    discretized_feature = []
    for feature in data.T:
        discretized_feature.append(list(np.linspace(0, 1, len(feature))))
    return np.array(discretized_feature)


def find_nearest(array, value):
    array = np.array(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


def find_sample_indices(discretized_descriptor, sample):
    indices = []
    for i, value in enumerate(sample):
        descriptor = np.asarray(discretized_descriptor[i])
        distances = np.abs(descriptor - value)
        indices.append(np.argmin(distances))
    return tuple(indices)


def get_row_index(input_data, column_indices):
    target_row = [input_data[column_index, feature_index] for feature_index, column_index in enumerate(column_indices)]
    target_row_np = np.array(target_row)
    matches = (input_data == target_row_np).all(axis=1)
    match_indices = np.where(matches)[0]
    return match_indices[0] if len(match_indices) > 0 else None


def random_sampling(n_sample, input_data):
    discretized_descriptor = discretize_dataset_sample(input_data)
    samples = set()
    while len(samples) < n_sample:
        sample = [random.choice(descriptor) for descriptor in discretized_descriptor]
        indices = find_sample_indices(discretized_descriptor, sample)
        row_index = get_row_index(input_data, indices)
        if row_index is not None:
            samples.add(row_index)
    return list(samples)
