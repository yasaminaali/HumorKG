"""Rubric-based single-output scorer.

Each joke gets four 1-5 scores from an impartial judge LLM:
    - funny:    does it actually amuse?
    - relevant: does it engage the input (headline topic / both words present)?
    - creative: original twist vs. generic?
    - fluent:   grammar / clarity in the target language?

Judge defaults to Groq's Llama-4-Scout so we stay within one free API. It's
also outside our generator set (Llama-3.1-8B, Llama-3.3-70B, Qwen3-32B,
GPT-OSS-120B), which reduces self-preference bias.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from typing import Literal

from openai import OpenAI

Lang = Literal["en", "es", "zh"]

RUBRIC_PROMPT = {
    "en": """You are an impartial humor judge. Given an input constraint (either two words that must both appear, or a news headline) and ONE candidate joke, score each dimension on a 1-5 scale (1=worst, 5=best):

- funny:    does it make you smile or laugh?
- relevant: for two words, are BOTH words present? for headlines, does it engage the topic?
- creative: is the twist original, or generic/clichéd?
- fluent:   is it grammatical and clear in English?

Return STRICT JSON only, no prose:
{"funny": int, "relevant": int, "creative": int, "fluent": int, "reason": "one short phrase"}""",
    "es": """Eres un juez imparcial de humor. Dado un input (dos palabras obligatorias o un titular) y UN chiste candidato, puntúa cada dimensión en escala 1-5 (1=peor, 5=mejor):

- funny:    ¿hace reír o sonreír?
- relevant: si es de dos palabras, ¿aparecen AMBAS? si es titular, ¿aborda el tema?
- creative: ¿el giro es original o tópico?
- fluent:   ¿gramática y claridad en español?

Devuelve JSON estricto, sin texto extra:
{"funny": int, "relevant": int, "creative": int, "fluent": int, "reason": "frase breve"}""",
    "zh": """你是公正的幽默评审员。给定一个输入（两个必须出现的词，或一条新闻标题）和一条候选笑话，对每个维度按 1-5 打分（1=最差，5=最好）：

- funny：是否让人笑或会心一笑
- relevant：两词是否都出现 / 是否切题
- creative：反转是否原创（而非老套）
- fluent：中文是否通顺

只输出严格 JSON，不要多余文字：
{"funny": int, "relevant": int, "creative": int, "fluent": int, "reason": "一句话"}""",
}


def _format_input(input_type: str, input_value: str, lang: Lang) -> str:
    if input_type == "words":
        parts = [p.strip() for p in input_value.split("|")]
        labels = {"en": "Required words", "es": "Palabras requeridas", "zh": "必须包含的词"}
        return f"{labels[lang]}: {', '.join(parts)}"
    labels = {"en": "Headline", "es": "Titular", "zh": "新闻标题"}
    return f"{labels[lang]}: {input_value}"


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


@dataclass
class RubricJudge:
    model: str = "meta-llama/llama-4-scout-17b-16e-instruct"
    api_key: str | None = None

    def __post_init__(self):
        self._client = OpenAI(
            api_key=self.api_key or os.environ.get("GROQ_API_KEY"),
            base_url="https://api.groq.com/openai/v1",
        )

    def score(self, input_type: str, input_value: str, output: str, lang: Lang, retries: int = 3) -> dict:
        system = RUBRIC_PROMPT[lang]
        user = f"{_format_input(input_type, input_value, lang)}\n\nCandidate joke:\n{output}\n\nReturn the JSON score now."
        last = None
        for attempt in range(retries):
            try:
                resp = self._client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    max_tokens=500,
                    temperature=0.0,
                )
                raw = resp.choices[0].message.content
                match = _JSON_RE.search(raw)
                if not match:
                    last = f"no JSON in: {raw[:120]}"
                    continue
                parsed = json.loads(match.group(0))
                for k in ("funny", "relevant", "creative", "fluent"):
                    v = int(parsed.get(k, 0))
                    parsed[k] = max(1, min(5, v))
                return parsed
            except Exception as exc:
                last = repr(exc)
                time.sleep(1.2 * (attempt + 1))
        return {"funny": 0, "relevant": 0, "creative": 0, "fluent": 0, "reason": f"judge_failed: {last}"}
