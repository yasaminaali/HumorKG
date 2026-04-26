"""Multilingual prompt builders.

Each prompt takes an (input_type, input_value, lang, kg_hint) and returns the
user prompt string to send to the LLM. If kg_hint is empty, the "optional
knowledge" section is omitted (clean baseline).
"""

from __future__ import annotations

MAX_CHARS = {"en": 900, "es": 900, "zh": 300}

_TWO_WORDS = {
    "en": 'Write ONE short joke (1-2 sentences) that MUST include both words: "{w1}" and "{w2}". '
          'Be original and concise. Avoid hate, harassment, slurs, explicit sexual content, or violence. '
          'Output only the joke itself — no preamble, no explanation.',
    "es": 'Escribe UN chiste corto (1-2 frases) que DEBE incluir estas dos palabras: "{w1}" y "{w2}". '
          'Sé original y conciso. Sin contenido ofensivo. '
          'Devuelve solo el chiste, sin preámbulo ni explicación.',
    "zh": '写一个简短的笑话（1-2句），必须同时包含这两个词：“{w1}” 和 “{w2}”。'
          '要原创、简洁，避免冒犯内容。只输出笑话本身，不要解释。',
}

_HEADLINE = {
    "en": 'Write ONE short joke (1-2 sentences) inspired by this news headline: "{h}". '
          'It must clearly relate to the headline. Be original; do not quote the article. '
          'Avoid hate, harassment, slurs, explicit sexual content, or violence. '
          'Output only the joke itself — no preamble, no explanation.',
    "es": 'Escribe UN chiste corto (1-2 frases) inspirado en este titular: "{h}". '
          'Debe relacionarse claramente con el titular. Sé original; no cites el artículo. '
          'Sin contenido ofensivo. Devuelve solo el chiste, sin preámbulo ni explicación.',
    "zh": '根据这条新闻标题写一个简短笑话（1-2句）：“{h}”。'
          '必须与标题相关，要原创，避免冒犯内容。只输出笑话本身，不要解释。',
}

_KG_PREFIX = {
    "en": "\n\nOptional knowledge (use if helpful):\n- {hint}",
    "es": "\n\nConocimiento opcional (usar si ayuda):\n- {hint}",
    "zh": "\n\n可选知识（如有帮助可使用）：\n- {hint}",
}


def build_prompt(input_type: str, input_value: str, lang: str, kg_hint: str = "") -> str:
    if input_type == "words":
        parts = [p.strip() for p in input_value.split("|")]
        w1, w2 = (parts + ["", ""])[:2]
        base = _TWO_WORDS[lang].format(w1=w1, w2=w2)
    elif input_type == "headline":
        base = _HEADLINE[lang].format(h=input_value)
    else:
        raise ValueError(f"unknown input_type: {input_type}")
    if kg_hint:
        base += _KG_PREFIX[lang].format(hint=kg_hint)
    return base


def classify_row(row: dict) -> tuple[str, str]:
    """Return (input_type, input_value) for a data-row dict."""
    w1, w2 = (row.get("word1") or "-").strip(), (row.get("word2") or "-").strip()
    if w1 and w1 != "-" and w2 and w2 != "-":
        return "words", f"{w1}|{w2}"
    h = (row.get("headline") or "").strip()
    if h and h != "-":
        return "headline", h
    raise ValueError(f"row {row.get('id')} has neither words nor headline")
