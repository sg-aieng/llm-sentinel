"""PII detection and redaction using a local token-classification model.

Same lifecycle and doctrine as the Prompt Guard classifier: loaded once
at startup, optional `[ml]` dependency, and graceful degradation — when
the model is unavailable, detect() returns None and the policy stage
downgrades the verdict to `review` instead of pretending the prompt is
PII-free.
"""
import logging
import re
from dataclasses import dataclass

from app.config import settings

logger = logging.getLogger("llm_sentinel.pii")

# Map model-specific entity groups to the canonical names used in policy
# files. Unmapped groups pass through uppercased.
_CANONICAL = {
    # piiranha-v1 labels
    "EMAIL": "EMAIL",
    "TELEPHONENUM": "PHONE",
    "SOCIALNUM": "SSN",
    "CREDITCARDNUMBER": "CREDIT_CARD",
    "GIVENNAME": "PERSON",
    "SURNAME": "PERSON",
    "USERNAME": "USERNAME",
    "PASSWORD": "PASSWORD",
    "ACCOUNTNUM": "ACCOUNT_NUMBER",
    "IDCARDNUM": "ID_NUMBER",
    "DRIVERLICENSENUM": "ID_NUMBER",
    "TAXNUM": "TAX_NUMBER",
    "DATEOFBIRTH": "DATE_OF_BIRTH",
    "STREET": "ADDRESS",
    "BUILDINGNUM": "ADDRESS",
    "CITY": "ADDRESS",
    "ZIPCODE": "ADDRESS",
    # dslim/bert-base-NER-style labels, if swapped in
    "PER": "PERSON",
    "LOC": "ADDRESS",
}


def _refine_span(text: str, start: int, end: int) -> tuple[int, int]:
    """Snap a model span to sensible boundaries.

    Token-classification spans are frequently sloppy: they include a
    leading space, or cover only part of a value ("john.smith" without
    "@example.com"). Trim whitespace, expand to whitespace-delimited
    token boundaries, then shed surrounding punctuation.
    """
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    while start > 0 and not text[start - 1].isspace():
        start -= 1
    while end < len(text) and not text[end].isspace():
        end += 1
    while end > start and text[end - 1] in ",.;:!?)]}\"'":
        end -= 1
    while start < end and text[start] in "([{\"'":
        start += 1
    return start, end


_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

# Fallback: scan for email addresses the ML model may miss because its
# tokenizer splits them (e.g. "alice" labelled GIVENNAME, "@example.com"
# not labelled at all).
_EMAIL_SCAN_RE = re.compile(r"\b[^\s@]+@[^\s@]+\.[^\s@]+\b")


def _entity_type(group: str, span_text: str) -> str:
    """Canonical type for a refined span.

    The model's label describes the tokens it saw, which for an email is
    often just the local part (labelled USERNAME). After _refine_span
    expands to the full address, retype by what the text actually is.
    """
    if _EMAIL_RE.match(span_text):
        return "EMAIL"
    return _CANONICAL.get(group.upper(), group.upper())


@dataclass
class PIIEntity:
    type: str  # canonical name, e.g. EMAIL
    text: str
    start: int
    end: int


class PIIDetector:
    def __init__(self, model_name: str | None = None, ner_model_name: str | None = None):
        self.model_name = model_name or settings.pii_model
        self.ner_model_name = (
            ner_model_name if ner_model_name is not None else settings.pii_ner_model
        )
        self.enabled = not settings.disable_pii
        self._pipeline = None
        self._ner_pipeline = None
        self._load_error: str | None = None

    def load(self) -> None:
        if not self.enabled:
            logger.info("PII detector disabled via SENTINEL_DISABLE_PII")
            return
        try:
            self._pipeline = self._load_pipeline(self.model_name)
            logger.info("loaded PII model %s", self.model_name)
        except Exception as exc:  # noqa: BLE001
            self._load_error = f"{type(exc).__name__}: {exc}"
            logger.warning("PII detector unavailable: %s", self._load_error)
            return

        # Secondary NER model for person names in prose, which the primary
        # model misses. Failure here degrades name coverage, not the stage.
        if self.ner_model_name:
            try:
                self._ner_pipeline = self._load_pipeline(self.ner_model_name)
                logger.info("loaded PII NER model %s", self.ner_model_name)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "PII NER model unavailable (person-name coverage reduced): %s",
                    exc,
                )

    @staticmethod
    def _load_pipeline(model_name: str):
        from transformers import (
            AutoModelForTokenClassification,
            AutoTokenizer,
            pipeline as hf_pipeline,
        )

        # Prefer the local cache; only touch the network on a cache miss
        # (first run). Avoids per-startup hub probes and works offline.
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
            model = AutoModelForTokenClassification.from_pretrained(
                model_name, local_files_only=True
            )
        except OSError:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForTokenClassification.from_pretrained(model_name)

        return hf_pipeline(
            "token-classification",
            model=model,
            tokenizer=tokenizer,
            aggregation_strategy="simple",
        )

    @property
    def available(self) -> bool:
        return self._pipeline is not None

    @property
    def load_error(self) -> str | None:
        return self._load_error

    def detect(self, text: str) -> list[PIIEntity] | None:
        """Return entities found, or None when the detector can't run."""
        if not self.enabled or self._pipeline is None:
            return None
        raw = list(self._pipeline(text))
        if self._ner_pipeline is not None:
            # Only take person/location groups from the general NER model;
            # ORG/MISC would over-redact.
            raw.extend(
                item for item in self._ner_pipeline(text)
                if item["entity_group"].upper() in ("PER", "LOC")
            )

        entities: list[PIIEntity] = []
        for item in sorted(raw, key=lambda i: i["start"]):
            start, end = _refine_span(text, int(item["start"]), int(item["end"]))
            if start >= end:
                continue
            # Model spans are often partial (e.g. just the local part of an
            # email). After expanding to token boundaries, drop any span
            # swallowed by the previous one.
            if entities and start < entities[-1].end:
                continue
            span_text = text[start:end]
            entities.append(
                PIIEntity(
                    type=_entity_type(item["entity_group"], span_text),
                    text=span_text,
                    start=start,
                    end=end,
                )
            )

        # Regex fallback: add any email addresses not already covered by an
        # ML span. The model tokenizer splits "alice@example.com" and may
        # label only the local part, missing the full address entirely.
        for m in _EMAIL_SCAN_RE.finditer(text):
            if not any(e.start <= m.start() < e.end for e in entities):
                entities.append(
                    PIIEntity(type="EMAIL", text=m.group(), start=m.start(), end=m.end())
                )

        entities.sort(key=lambda e: e.start)
        return entities


def redact(
    text: str, entities: list[PIIEntity], types: list[str] | None = None
) -> tuple[str, list[str]]:
    """Replace entity spans with stable placeholders like [EMAIL_1].

    The same source text always maps to the same placeholder within one
    prompt, so the LLM can still refer to redacted values coherently.
    `types` limits which canonical types are redacted (None = all).
    Returns (redacted_text, sorted list of types found among targets).
    """
    targets = [
        e for e in entities if types is None or e.type in types
    ]
    if not targets:
        return text, []

    placeholders: dict[tuple[str, str], str] = {}
    counters: dict[str, int] = {}
    for entity in targets:
        key = (entity.type, entity.text)
        if key not in placeholders:
            counters[entity.type] = counters.get(entity.type, 0) + 1
            placeholders[key] = f"[{entity.type}_{counters[entity.type]}]"

    # Replace from the end so earlier offsets stay valid.
    result = text
    for entity in sorted(targets, key=lambda e: e.start, reverse=True):
        result = (
            result[: entity.start]
            + placeholders[(entity.type, entity.text)]
            + result[entity.end:]
        )
    return result, sorted({e.type for e in targets})
