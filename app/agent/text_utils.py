"""Lightweight Arabic normalization and plain-text sanitization for the Conversational AI Agent."""

from __future__ import annotations

import re

ARABIC_INDIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")

URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
MARKDOWN_LINK_PATTERN = re.compile(r"\[([^\]]+)\]\([^)]+\)")
MARKDOWN_CHARS_PATTERN = re.compile(r"[*_~`#>+]")
EMOJI_PATTERN = re.compile(
    r"[\U00010000-\U0010ffff]|"
    r"[\u2600-\u26ff\u2700-\u27bf\u2300-\u23ff\u2b50\u200d\ufe0f]",
    flags=re.UNICODE,
)
MULTIPLE_SPACES = re.compile(r"\s+")

TASHKEEL_PATTERN = re.compile(r"[\u064B-\u0652\u0670]")
TATWEEL_PATTERN = re.compile(r"\u0640")
PUNCTUATION_PATTERN = re.compile(r"[!\"#$%&\'()*+,-./:;<=>?@\[\\\]^_`{|}~،؛؟«»—–…\n\r\t]")


def normalize_arabic(text: str) -> str:
    """Standardize Arabic text for deterministic keyword and FAQ matching."""
    if not text:
        return ""

    normalized = TASHKEEL_PATTERN.sub("", text)
    normalized = TATWEEL_PATTERN.sub("", normalized)
    normalized = re.sub(r"[أإآٱ]", "ا", normalized)
    normalized = re.sub(r"ى", "ي", normalized)
    normalized = re.sub(r"ة", "ه", normalized)
    normalized = normalized.translate(ARABIC_INDIC_DIGITS)
    normalized = PUNCTUATION_PATTERN.sub(" ", normalized)
    normalized = normalized.lower()
    return MULTIPLE_SPACES.sub(" ", normalized).strip()
