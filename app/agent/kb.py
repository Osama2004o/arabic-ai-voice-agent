"""Fast local Arabic FAQ knowledge-base loader and retriever (< 2 ms per query)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

from .text_utils import normalize_arabic

_KB_PATH = Path(__file__).resolve().parent / "knowledge_base.json"

_ARABIC_STOPWORDS = frozenset({
    "من", "في", "على", "الى", "عن", "مع", "هل", "ما", "هو", "هي", "انا", "انت",
    "ده", "دي", "لو", "او", "و", "ب", "ل", "ك", "ال", "في", "ان", "كان", "كل",
    "ايه", "ازاي", "كام", "امتى", "فين", "ليه", "ممكن", "عايز", "عايزة", "لو سمحت",
    "اريد", "اود", "كيف", "متى", "كم", "اين", "لماذا", "من فضلك", "عندي", "لدي",
})


@dataclass(frozen=True)
class FAQEntry:
    id: str
    category: str
    question: str
    keywords: Tuple[str, ...]
    answer: str
    norm_question: str
    norm_keywords: Tuple[str, ...]
    tokens: frozenset[str]


def _tokenize(text: str) -> frozenset[str]:
    norm = normalize_arabic(text)
    tokens = {
        tok for tok in norm.split()
        if len(tok) > 1 and tok not in _ARABIC_STOPWORDS
    }
    return frozenset(tokens)


def _load_entries(path: Path = _KB_PATH) -> Tuple[FAQEntry, ...]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    entries: List[FAQEntry] = []
    for item in raw:
        keywords = tuple(item.get("keywords", []))
        norm_keywords = tuple(normalize_arabic(k) for k in keywords)
        combined = " ".join([item["question"], *keywords, item.get("category", "")])
        entries.append(
            FAQEntry(
                id=item["id"],
                category=item["category"],
                question=item["question"],
                keywords=keywords,
                answer=item["answer"],
                norm_question=normalize_arabic(item["question"]),
                norm_keywords=norm_keywords,
                tokens=_tokenize(combined),
            )
        )
    return tuple(entries)


_FAQ_ENTRIES: Tuple[FAQEntry, ...] = _load_entries()


def get_all_faqs() -> Tuple[FAQEntry, ...]:
    """Return all cached FAQ entries."""
    return _FAQ_ENTRIES


def retrieve_faq(query: str, top_k: int = 3, min_score: float = 0.12) -> List[Tuple[FAQEntry, float]]:
    """
    Score and return the top-k matching FAQ entries for an Arabic query.
    Combines exact normalized keyword phrase hits with token overlap so both
    MSA and Egyptian colloquial phrasing match cleanly in < 2 ms.
    """
    if not query or not query.strip():
        return []

    norm_query = normalize_arabic(query)
    query_tokens = _tokenize(query)

    scored: List[Tuple[FAQEntry, float]] = []
    for entry in _FAQ_ENTRIES:
        phrase_hits = sum(1 for kw in entry.norm_keywords if kw and kw in norm_query)
        token_overlap = len(query_tokens & entry.tokens)

        # Prefix/stem-friendly soft match (handles Arabic prefixes like بـ / الـ / وـ)
        soft_overlap = 0
        if query_tokens and entry.tokens:
            for qt in query_tokens:
                qt_core = qt[2:] if qt.startswith("ال") and len(qt) > 3 else qt
                for et in entry.tokens:
                    et_core = et[2:] if et.startswith("ال") and len(et) > 3 else et
                    if qt_core == et_core or (len(qt_core) >= 4 and qt_core in et_core):
                        soft_overlap += 1
                        break

        denom = max(len(query_tokens), 1)
        token_score = (token_overlap + 0.6 * max(soft_overlap - token_overlap, 0)) / denom
        phrase_score = min(phrase_hits * 0.45, 1.35)
        score = round(phrase_score + token_score, 4)

        if score >= min_score:
            scored.append((entry, score))

    scored.sort(key=lambda pair: pair[1], reverse=True)
    return scored[:top_k]


def format_faq_context(matches: List[Tuple[FAQEntry, float]]) -> str:
    """Format retrieved FAQ entries as compact context for the system prompt."""
    if not matches:
        return "لا توجد معلومة مطابقة مباشرة في قاعدة المعرفة لهذا السؤال."

    lines = []
    for entry, _score in matches:
        lines.append(f"- [{entry.id}] س: {entry.question} | ج: {entry.answer}")
    return "\n".join(lines)
