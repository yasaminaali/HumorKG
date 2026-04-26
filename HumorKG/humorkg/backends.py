"""LLM backends: one API per provider, one protocol.

All backends expose `.generate(prompt, max_new_tokens, temperature, top_p) -> str`
and return ONLY the assistant's text, with no role-token prefix. This is the
fix for the "assistant" leakage in the Colab script.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Protocol


class LLMBackend(Protocol):
    name: str
    model: str
    def generate(self, prompt: str, max_new_tokens: int = 120, temperature: float = 0.7, top_p: float = 0.9) -> str: ...


def _openai_compatible_generate(client, model: str, prompt: str, max_new_tokens: int, temperature: float, top_p: float) -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
    )
    return resp.choices[0].message.content.strip()


@dataclass
class TogetherBackend:
    model: str = "meta-llama/Llama-3.3-70B-Instruct-Turbo"
    api_key: str | None = None
    name: str = "together"

    def __post_init__(self):
        from openai import OpenAI
        self._client = OpenAI(
            api_key=self.api_key or os.environ.get("TOGETHER_API_KEY"),
            base_url="https://api.together.xyz/v1",
        )

    def generate(self, prompt, max_new_tokens=120, temperature=0.7, top_p=0.9) -> str:
        return _openai_compatible_generate(self._client, self.model, prompt, max_new_tokens, temperature, top_p)


@dataclass
class GroqBackend:
    model: str = "llama-3.3-70b-versatile"
    api_key: str | None = None
    name: str = "groq"

    def __post_init__(self):
        from openai import OpenAI
        self._client = OpenAI(
            api_key=self.api_key or os.environ.get("GROQ_API_KEY"),
            base_url="https://api.groq.com/openai/v1",
        )

    def generate(self, prompt, max_new_tokens=120, temperature=0.7, top_p=0.9) -> str:
        # Qwen3 defaults to chain-of-thought. For joke generation, CoT just
        # burns tokens — disable it with the /no_think directive.
        if "qwen3" in self.model.lower():
            prompt = prompt + " /no_think"
        return _openai_compatible_generate(self._client, self.model, prompt, max_new_tokens, temperature, top_p)


@dataclass
class AnthropicBackend:
    model: str = "claude-opus-4-7"
    api_key: str | None = None
    name: str = "anthropic"

    def __post_init__(self):
        from anthropic import Anthropic
        self._client = Anthropic(api_key=self.api_key or os.environ.get("ANTHROPIC_API_KEY"))

    def generate(self, prompt, max_new_tokens=120, temperature=0.7, top_p=0.9) -> str:
        resp = self._client.messages.create(
            model=self.model,
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text.strip()


@dataclass
class HFLocalBackend:
    """Reproduces the Colab setup correctly. Requires torch + transformers + a GPU."""
    model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    quantize_4bit: bool = True
    name: str = "hf_local"

    def __post_init__(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(self.model, use_fast=True)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        kwargs = {"device_map": "auto"}
        if self.quantize_4bit and torch.cuda.is_available():
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        else:
            kwargs["torch_dtype"] = torch.float32

        self._model = AutoModelForCausalLM.from_pretrained(self.model, **kwargs)
        self._model.eval()

    def generate(self, prompt, max_new_tokens=120, temperature=0.7, top_p=0.9) -> str:
        messages = [{"role": "user", "content": prompt}]
        input_ids = self._tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ).to(self._model.device)

        with self._torch.no_grad():
            output_ids = self._model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=1.05,
                pad_token_id=self._tokenizer.eos_token_id,
            )

        # Slice off the prompt by length — correct way, avoids the "assistant"
        # role-token leakage that the Colab script's string-split logic produced.
        gen_ids = output_ids[0][input_ids.shape[-1]:]
        return self._tokenizer.decode(gen_ids, skip_special_tokens=True).strip()


BACKENDS = {
    "together": TogetherBackend,
    "groq": GroqBackend,
    "anthropic": AnthropicBackend,
    "hf_local": HFLocalBackend,
}


def build_backend(name: str, model: str | None = None) -> LLMBackend:
    cls = BACKENDS[name]
    return cls(model=model) if model else cls()


def generate_with_retry(backend: LLMBackend, prompt: str, retries: int = 8, **kwargs) -> str:
    last = None
    for attempt in range(retries):
        try:
            return backend.generate(prompt, **kwargs)
        except Exception as exc:
            last = exc
            # Exponential backoff capped at 30s.
            time.sleep(min(30.0, 2.0 * (2 ** attempt)))
    raise RuntimeError(f"backend {backend.name} failed after {retries} retries: {last}")
