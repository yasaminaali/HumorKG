"""Main generation loop for one (backend × kg × language) cell.

Produces a TSV in submission format: `id<tab>text`. Clips to the per-language
character limit. Writes a `.partial` checkpoint every N rows so a crashed run
can resume.
"""

from __future__ import annotations

import csv
import os
import re
import sys
import unicodedata
from pathlib import Path

from humorkg.backends import LLMBackend, generate_with_retry
from humorkg.kg import KG
from humorkg.prompts import MAX_CHARS, build_prompt, classify_row


def _normalize(text: str, max_chars: int) -> str:
    text = unicodedata.normalize("NFKC", str(text))
    # Strip reasoning-model "think" blocks (Qwen3, DeepSeek-R1 etc).
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    # If a think block was truncated (no closing tag), drop everything up to the
    # last sentence-like boundary to salvage the final answer.
    if "<think>" in text.lower():
        # Keep only content after the last "</think>" or, if none, give up on this output.
        parts = re.split(r"</think>", text, flags=re.IGNORECASE)
        text = parts[-1] if len(parts) > 1 else ""
    text = re.sub(r"\s+", " ", text).strip()
    # Strip any "assistant" leakage that a naive backend might still produce.
    text = re.sub(r"^(assistant\s*[:\-]?\s*)+", "", text, flags=re.IGNORECASE)
    return text[:max_chars]


def _read_tsv(path: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def _write_tsv(path: str, rows: list[dict]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "text"], delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _append_partial(path: str, row: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    header = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "text"], delimiter="\t", extrasaction="ignore")
        if header:
            writer.writeheader()
        writer.writerow(row)


def _load_done_ids(partial_path: str) -> set:
    if not os.path.exists(partial_path):
        return set()
    return {r["id"] for r in _read_tsv(partial_path)}


def run(
    backend: LLMBackend,
    kg: KG,
    input_path: str,
    output_path: str,
    lang: str,
    max_new_tokens: int = 400,
    temperature: float = 0.7,
    top_p: float = 0.9,
    limit: int | None = None,
) -> None:
    max_chars = MAX_CHARS[lang]
    partial = output_path + ".partial"
    done = _load_done_ids(partial)

    rows = _read_tsv(input_path)
    if limit:
        rows = rows[:limit]
    if done:
        print(f"resuming; {len(done)} rows already done", flush=True)

    for i, row in enumerate(rows):
        rid = row["id"]
        if rid in done:
            continue
        input_type, input_value = classify_row(row)
        kg_hint = kg.hint(input_type, input_value, lang)
        prompt = build_prompt(input_type, input_value, lang, kg_hint)

        raw = generate_with_retry(backend, prompt, max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p)
        text = _normalize(raw, max_chars)
        _append_partial(partial, {"id": rid, "text": text})
        done.add(rid)
        if (i + 1) % 10 == 0 or i + 1 == len(rows):
            print(f"[{backend.name}:{backend.model} | kg={kg.name} | {lang}] {i + 1}/{len(rows)}", flush=True)

    # Merge to final, preserving original row order
    partial_rows = {r["id"]: r["text"] for r in _read_tsv(partial)}
    fallback = {"en": "I tried to write a joke, but the punchline missed its connection.",
                "es": "Intenté escribir un chiste, pero el remate perdió la conexión.",
                "zh": "我想写个笑话，但笑点没赶上。"}[lang]
    final = [{"id": r["id"], "text": _normalize(partial_rows.get(r["id"], fallback), max_chars)} for r in rows]
    _write_tsv(output_path, final)
    print(f"wrote {output_path} ({len(final)} rows)", flush=True)
