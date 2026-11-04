"""Pipeline stage 3 — deterministic rule checks.

Regex and heuristic checks against the normalized prompt. Any violation
blocks the request; the classifier never runs when a rule fires.

All patterns and thresholds come from app.config (env / .env file).
Rule patterns themselves live in the YAML rule pack (rules_file setting).
"""
import re
from dataclasses import dataclass, field
from importlib.resources import files as _resource_files
from pathlib import Path

import yaml

from app.config import settings

_BASE64_BLOB = re.compile(settings.base64_blob_pattern)
_HEX_BLOB = re.compile(settings.hex_blob_pattern)


@dataclass
class CompiledRule:
    id: str
    category: str
    description: str
    pattern: re.Pattern


@dataclass
class RuleCheckResult:
    passed: bool
    violations: list[str] = field(default_factory=list)


class RuleEngine:
    def __init__(self, rules_path: Path | None = None):
        # When no explicit path is given, resolve via importlib.resources so the
        # engine works correctly in both source trees and installed wheels.
        if rules_path is None:
            self._resource = _resource_files("app") / "rules" / settings.rules_file
        else:
            self._resource = rules_path
        self.max_encoded_ratio = settings.max_encoded_ratio
        self.min_blob_length = settings.min_blob_length
        self.rules: list[CompiledRule] = []
        self._load()

    def _load(self) -> None:
        raw = yaml.safe_load(self._resource.read_text(encoding="utf-8"))
        # Any top-level `*_patterns` key is a rule category, so the same
        # engine serves both the input and output rule packs.
        for category in (k for k in raw if k.endswith("_patterns")):
            for entry in raw.get(category, []) or []:
                self.rules.append(
                    CompiledRule(
                        id=entry["id"],
                        category=category,
                        description=entry.get("description", ""),
                        pattern=re.compile(entry["pattern"], re.IGNORECASE),
                    )
                )

    def check(self, text: str) -> RuleCheckResult:
        violations: list[str] = []

        for rule in self.rules:
            if rule.pattern.search(text):
                violations.append(rule.id)

        if self._encoded_ratio(text) > self.max_encoded_ratio:
            violations.append("excessive_encoded_content")

        return RuleCheckResult(passed=not violations, violations=violations)

    def _encoded_ratio(self, text: str) -> float:
        if not text:
            return 0.0
        encoded_chars = 0
        for pattern in (_BASE64_BLOB, _HEX_BLOB):
            for match in pattern.finditer(text):
                if len(match.group()) >= self.min_blob_length:
                    encoded_chars += len(match.group())
        return min(encoded_chars / len(text), 1.0)
