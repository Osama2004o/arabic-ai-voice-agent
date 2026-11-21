import re

# Arabic-Indic to ASCII digits mapping
ARABIC_INDIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")

# Regex patterns for cleaning text for TTS
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
MARKDOWN_LINK_PATTERN = re.compile(r"\[([^\]]+)\]\([^)]+\)")
MARKDOWN_CHARS_PATTERN = re.compile(r"[*_~`#>+]")
EMOJI_PATTERN = re.compile(
    r"[\U00010000-\U0010ffff]|"
    r"[\u2600-\u26ff\u2700-\u27bf\u2300-\u23ff\u2b50\u200d\ufe0f]",
    flags=re.UNICODE,
)
MULTIPLE_SPACES = re.compile(r"\s+")

# Regex patterns for Arabic normalization (WER calculation)
TASHKEEL_PATTERN = re.compile(r"[\u064B-\u0652\u0670]")
TATWEEL_PATTERN = re.compile(r"\u0640")
PUNCTUATION_PATTERN = re.compile(r"[!\"#$%&\'()*+,-./:;<=>?@\[\\\]^_`{|}~،؛؟«»—–…\n\r\t]")


def clean_text_for_tts(text: str, max_chars: int = 1000) -> str:
    """
    Cleans agent reply text to optimize for natural and low-latency Arabic TTS.
    - Strips markdown formatting (bold, italic, headers, backticks, bullets)
    - Replaces markdown links [text](url) with just text
    - Removes raw URLs and emojis
    - Spaces out long numbers (>3 digits) so Arabic TTS reads individual digits clearly
    - Collapses whitespace and caps maximum length to protect latency
    """
    if not text:
        return ""

    # Replace markdown links with their visible label
    cleaned = MARKDOWN_LINK_PATTERN.sub(r"\1", text)

    # Remove URLs
    cleaned = URL_PATTERN.sub("", cleaned)

    # Remove markdown formatting characters
    cleaned = MARKDOWN_CHARS_PATTERN.sub("", cleaned)

    # Remove emojis and miscellaneous symbols
    cleaned = EMOJI_PATTERN.sub("", cleaned)

    # Space out long digit sequences (>3 digits, e.g., order # 12485 -> 1 2 4 8 5)
    def _space_digits(match: re.Match) -> str:
        s = match.group(0)
        return " ".join(s) if len(s) > 3 else s

    cleaned = re.sub(r"\b\d+\b", _space_digits, cleaned)

    # Collapse whitespace
    cleaned = MULTIPLE_SPACES.sub(" ", cleaned).strip()

    # Cap length to avoid excessive TTS delay
    if len(cleaned) > max_chars:
        # Find nearest space before limit if possible
        cut = cleaned[:max_chars]
        last_space = cut.rfind(" ")
        if last_space > int(max_chars * 0.8):
            cleaned = cut[:last_space].strip() + "."
        else:
            cleaned = cut.strip() + "."

    return cleaned


def normalize_arabic(text: str) -> str:
    """
    Standardizes Arabic text strictly for WER (Word Error Rate) scoring.
    - Removes diacritics (tashkeel) and tatweel (kashida)
    - Unifies Alef variants (أ / إ / آ / ٱ -> ا)
    - Unifies Ya variants (ى -> ي)
    - Unifies Teh Marbuta (ة -> ه)
    - Converts Arabic-Indic digits to standard ASCII digits
    - Removes all Arabic and English punctuation
    - Lowers ASCII case and collapses whitespace
    """
    if not text:
        return ""

    # Remove Tashkeel and Tatweel
    normalized = TASHKEEL_PATTERN.sub("", text)
    normalized = TATWEEL_PATTERN.sub("", normalized)

    # Unify Alef forms
    normalized = re.sub(r"[أإآٱ]", "ا", normalized)

    # Unify Ya forms
    normalized = re.sub(r"ى", "ي", normalized)

    # Unify Teh Marbuta
    normalized = re.sub(r"ة", "ه", normalized)

    # Convert Arabic-Indic numerals
    normalized = normalized.translate(ARABIC_INDIC_DIGITS)

    # Strip punctuation
    normalized = PUNCTUATION_PATTERN.sub(" ", normalized)

    # Lowercase Latin characters
    normalized = normalized.lower()

    # Collapse multiple whitespaces
    normalized = MULTIPLE_SPACES.sub(" ", normalized).strip()

    return normalized
