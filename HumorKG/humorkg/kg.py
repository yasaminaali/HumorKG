"""KG retrieval adapters.

Each KG exposes `.hint(input_type, input_value, lang) -> str` that returns a
short bulleted hint string ready to splice into the prompt, or "" if nothing.

- NoKG: returns "". For the ablation baseline.
- WordNetKG: English only (via NLTK). Hypernyms + antonyms for two-word inputs,
  headline keywords for headlines. Exactly matches what `humorai .py` did.
- ConceptNetKG: multilingual REST API. Edges per term in target language,
  prefers concept overlap / IsA / RelatedTo for anchor candidates.
"""

from __future__ import annotations

import re
import time
from functools import lru_cache
from typing import Protocol


class KG(Protocol):
    name: str
    def hint(self, input_type: str, input_value: str, lang: str) -> str: ...


class NoKG:
    name = "none"
    def hint(self, input_type: str, input_value: str, lang: str) -> str:
        return ""


_STOP = set(("the a an and or to of in on for with is are was were be as at by from that this it "
             "into no not if then while i you we they he she them his her our your their").split())


def _tokenize(text: str) -> list[str]:
    words = re.findall(r"[a-zA-Z']+", (text or "").lower())
    return [w for w in words if w not in _STOP and len(w) > 2]


class WordNetKG:
    name = "wordnet"

    def __init__(self):
        import nltk
        try:
            nltk.data.find("corpora/wordnet")
        except LookupError:
            nltk.download("wordnet", quiet=True)
            nltk.download("omw-1.4", quiet=True)
        from nltk.corpus import wordnet as wn
        self._wn = wn

    def _relations_for(self, word: str, maxn: int = 3) -> list[str]:
        rels = set()
        for syn in self._wn.synsets(word)[:2]:
            for h in syn.hypernyms():
                rels.add(f"{word} is a kind of {h.lemmas()[0].name().replace('_', ' ')}")
            for lem in syn.lemmas():
                for ant in lem.antonyms():
                    rels.add(f"{word} contrasts with {ant.name().replace('_', ' ')}")
        return list(rels)[:maxn]

    def hint(self, input_type: str, input_value: str, lang: str) -> str:
        if input_type == "words":
            w1, w2 = [p.strip() for p in input_value.split("|")[:2]]
            rels = self._relations_for(w1) + self._relations_for(w2)
            return "; ".join(rels)
        # headline: take top keywords and look each up
        kws = _tokenize(input_value)[:5]
        rels = []
        for w in kws:
            rels += self._relations_for(w, maxn=2)
        return "; ".join(rels[:6])


@lru_cache(maxsize=8192)
def _conceptnet_edges(lang: str, term: str, limit: int = 40) -> tuple:
    import requests
    t = re.sub(r"\s+", "_", term.strip().lower())
    t = re.sub(r"[^a-z0-9_'-\u00C0-\uFFFF]", "", t)
    if not t:
        return ()
    url = f"https://api.conceptnet.io/c/{lang}/{t}?limit={limit}"
    for attempt in range(3):
        try:
            r = requests.get(url, timeout=15)
            if r.status_code == 200:
                data = r.json()
                out = []
                for e in data.get("edges", []):
                    rel = e["rel"]["label"]
                    start = e["start"]["label"]
                    end = e["end"]["label"]
                    start_lang = e["start"].get("language", "")
                    end_lang = e["end"].get("language", "")
                    weight = float(e.get("weight", 1.0))
                    out.append((rel, start, end, start_lang, end_lang, weight))
                return tuple(out)
            if 500 <= r.status_code < 600:
                time.sleep(0.8 * (attempt + 1))
                continue
            return ()
        except Exception:
            time.sleep(0.8 * (attempt + 1))
    return ()


def _neighbors(edges: tuple, focus: str, lang: str, maxn: int = 8):
    focus = focus.lower()
    out = []
    for rel, start, end, sl, el, w in edges:
        if start.lower() == focus and el == lang:
            out.append((end, rel, w))
        elif end.lower() == focus and sl == lang:
            out.append((start, rel, w))
    out.sort(key=lambda x: x[2], reverse=True)
    return out[:maxn]


class ConceptNetKG:
    name = "conceptnet"

    def hint(self, input_type: str, input_value: str, lang: str) -> str:
        if input_type == "words":
            w1, w2 = [p.strip() for p in input_value.split("|")[:2]]
            e1 = _conceptnet_edges(lang, w1)
            e2 = _conceptnet_edges(lang, w2)
            n1 = _neighbors(e1, w1, lang, maxn=10)
            n2 = _neighbors(e2, w2, lang, maxn=10)
            set1 = {n[0].lower() for n in n1}
            set2 = {n[0].lower() for n in n2}
            overlap = sorted(set1 & set2)[:5]
            hints = []
            if overlap:
                hints.append(f"shared concepts: {', '.join(overlap)}")
            if n1:
                hints.append(f"{w1}: {', '.join(f'{n[0]} ({n[1]})' for n in n1[:4])}")
            if n2:
                hints.append(f"{w2}: {', '.join(f'{n[0]} ({n[1]})' for n in n2[:4])}")
            return "; ".join(hints)
        # headline: look up top-3 tokens
        toks = _tokenize(input_value) if lang == "en" else _cjk_tokens(input_value) if lang == "zh" else _latin_tokens(input_value)
        toks = toks[:3]
        hints = []
        for t in toks:
            edges = _conceptnet_edges(lang, t)
            n = _neighbors(edges, t, lang, maxn=4)
            if n:
                hints.append(f"{t}: {', '.join(x[0] for x in n)}")
        return "; ".join(hints)


def _cjk_tokens(text: str) -> list[str]:
    # crude: emit each CJK char / run as a token candidate, then fall back to 2-gram
    chars = [c for c in (text or "") if "\u4e00" <= c <= "\u9fff"]
    if not chars:
        return []
    bigrams = ["".join(chars[i:i+2]) for i in range(len(chars) - 1)]
    return list(dict.fromkeys(bigrams))


def _latin_tokens(text: str) -> list[str]:
    words = re.findall(r"[A-Za-zÀ-ÿ]+", (text or "").lower())
    return [w for w in words if len(w) > 3][:10]


KGS = {"none": NoKG, "wordnet": WordNetKG, "conceptnet": ConceptNetKG}


def build_kg(name: str) -> KG:
    return KGS[name]()
