"""Pipeline stage 2 — normalization.

Cleans the raw prompt before rule checks and classification. The original
prompt is hashed by the caller before this stage runs, so nothing is lost
for audit purposes.

All patterns and thresholds are compiled once at import time from app.config.
"""
import html
import re
import unicodedata
from dataclasses import dataclass, field

from app.config import settings

_INVISIBLE = re.compile(f"[{settings.invisible_chars_pattern}]")
_SPACE_RUNS = re.compile(r"[ \t]{" + str(settings.max_space_run) + r",}")
_NEWLINE_RUNS = re.compile(r"\n{" + str(settings.max_newline_run) + r",}")


@dataclass
class NormalizationResult:
    text: str
    applied: list[str] = field(default_factory=list)


def normalize(prompt: str) -> NormalizationResult:
    applied: list[str] = []
    text = prompt

    nfc = unicodedata.normalize("NFC", text)
    if nfc != text:
        applied.append("unicode_nfc")
    text = nfc

    stripped = _INVISIBLE.sub("", text)
    if stripped != text:
        applied.append("strip_invisible_chars")
    text = stripped

    # Decode HTML entities exactly once — repeated decoding would itself
    # be an obfuscation vector.
    unescaped = html.unescape(text)
    if unescaped != text:
        applied.append("html_entity_decode")
    text = unescaped

    collapsed = _SPACE_RUNS.sub(" ", text)
    collapsed = _NEWLINE_RUNS.sub("\n\n", collapsed)
    if collapsed != text:
        applied.append("collapse_whitespace")
    text = collapsed

    trimmed = text.strip()
    if trimmed != text:
        applied.append("trim")
    text = trimmed

    return NormalizationResult(text=text, applied=applied)
