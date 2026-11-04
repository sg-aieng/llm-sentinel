"""Pipeline stage 5 — policy evaluation.

Runs only when the request survived rule checks and the classifier, and
only when the caller supplied an application_id. Checks, in order:

  1. Capabilities — requested capabilities against the policy's
     allowed/blocked lists (role-scoped lists override the defaults).
  2. Topics — keyword blocklist against the normalized prompt.
  3. PII — ML detection, then the policy's action: block, redact, flag.

Policy can only hold or downgrade a verdict (allow → review → block),
never upgrade one. An application_id with no policy file gets the
strict default: no capabilities allowed, PII flagged.
"""
import logging
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from app.config import settings
from app.pipeline.pii import PIIDetector, redact

logger = logging.getLogger("llm_sentinel.policy")

_VALID_PII_ACTIONS = {"block", "redact", "flag"}

STRICT_DEFAULT_POLICY = {
    "capabilities": {"allowed": [], "blocked": []},
    "topics": {"blocked": []},
    "pii": {"action": "flag", "entities": None},
}


@dataclass
class PolicyDecision:
    evaluated: bool = False
    verdict: str | None = None  # block | review | None (no change)
    blocked_by: str | None = None  # capabilities | topics | pii
    capability_violations: list[str] = field(default_factory=list)
    topic_matches: list[str] = field(default_factory=list)
    pii_found: list[str] = field(default_factory=list)
    pii_action: str | None = None
    redacted_prompt: str | None = None
    note: str | None = None


class PolicyEngine:
    def __init__(self, policies_dir: Path | None = None):
        self.policies_dir = policies_dir or (
            Path(__file__).resolve().parents[2] / settings.policies_dir
        )
        self.policies: dict[str, dict] = {}
        self._load()

    _KNOWN_KEYS = {"id", "capabilities", "topics", "pii", "roles"}

    def _load(self) -> None:
        if not self.policies_dir.is_dir():
            logger.info("no policies directory at %s", self.policies_dir)
            return
        for path in sorted(self.policies_dir.glob("*.yaml")):
            doc = yaml.safe_load(path.read_text()) or {}
            self._validate_policy(path.stem, doc)
            app_id = doc.get("id") or path.stem
            self.policies[app_id] = doc
        logger.info("loaded %d policies from %s", len(self.policies), self.policies_dir)

    def _validate_policy(self, name: str, doc: object) -> None:
        """Validate a policy document, raising ValueError on any structural error."""
        if not isinstance(doc, dict):
            raise ValueError(f"policy {name!r}: must be a YAML mapping, got {type(doc).__name__}")

        unknown = set(doc) - self._KNOWN_KEYS
        if unknown:
            raise ValueError(f"policy {name!r}: unknown top-level keys {sorted(unknown)}")

        # Validate id type
        if "id" in doc and not isinstance(doc["id"], str):
            raise ValueError(
                f"policy {name!r}: 'id' must be a string, got {type(doc['id']).__name__}"
            )

        # Validate capabilities section
        caps_section = doc.get("capabilities")
        if caps_section is not None:
            self._validate_capabilities(name, "capabilities", caps_section)

        # Validate roles section
        roles_section = doc.get("roles")
        if roles_section is not None:
            if not isinstance(roles_section, dict):
                raise ValueError(f"policy {name!r}: 'roles' must be a mapping")
            for role_name, role_def in roles_section.items():
                if not isinstance(role_name, str):
                    raise ValueError(
                        f"policy {name!r}: role name must be a string, got {type(role_name).__name__}"
                    )
                if not isinstance(role_def, dict):
                    raise ValueError(
                        f"policy {name!r}: roles[{role_name!r}] must be a mapping"
                    )
                unknown_role_keys = set(role_def) - {"capabilities"}
                if unknown_role_keys:
                    raise ValueError(
                        f"policy {name!r}: roles[{role_name!r}] has unknown keys "
                        f"{sorted(unknown_role_keys)}"
                    )
                role_caps = role_def.get("capabilities")
                if role_caps is not None:
                    self._validate_capabilities(
                        name, f"roles[{role_name!r}].capabilities", role_caps
                    )

        # Validate PII section
        pii_section = doc.get("pii") or {}
        if pii_section:
            if not isinstance(pii_section, dict):
                raise ValueError(f"policy {name!r}: 'pii' must be a mapping")
            action = pii_section.get("action")
            if action is not None and action not in _VALID_PII_ACTIONS:
                raise ValueError(
                    f"policy {name!r}: pii.action must be one of "
                    f"{sorted(_VALID_PII_ACTIONS)}, got {action!r}"
                )
            entities = pii_section.get("entities")
            if entities is not None:
                if not isinstance(entities, list):
                    raise ValueError(
                        f"policy {name!r}: pii.entities must be a list, "
                        f"got {type(entities).__name__}"
                    )
                for i, e in enumerate(entities):
                    if not isinstance(e, str):
                        raise ValueError(
                            f"policy {name!r}: pii.entities[{i}] must be a string, "
                            f"got {type(e).__name__}"
                        )

        # Validate topics section
        topics_section = doc.get("topics") or {}
        if topics_section and not isinstance(topics_section, dict):
            raise ValueError(f"policy {name!r}: 'topics' must be a mapping")
        for i, topic in enumerate((topics_section.get("blocked") or [])):
            if not isinstance(topic, dict):
                raise ValueError(
                    f"policy {name!r}: topics.blocked[{i}] must be a mapping"
                )
            topic_id = topic.get("id")
            if topic_id is None:
                raise ValueError(
                    f"policy {name!r}: topics.blocked[{i}] missing required key 'id'"
                )
            if not isinstance(topic_id, str):
                raise ValueError(
                    f"policy {name!r}: topics.blocked[{i}].id must be a string, "
                    f"got {type(topic_id).__name__}"
                )
            kws = topic.get("keywords")
            if not isinstance(kws, list) or not kws:
                raise ValueError(
                    f"policy {name!r}: topics.blocked[{i}].keywords must be a non-empty list"
                )
            for j, kw in enumerate(kws):
                if not isinstance(kw, str):
                    raise ValueError(
                        f"policy {name!r}: topics.blocked[{i}].keywords[{j}] must be a string, "
                        f"got {type(kw).__name__}"
                    )

    def _validate_capabilities(self, name: str, path: str, caps: object) -> None:
        if not isinstance(caps, dict):
            raise ValueError(
                f"policy {name!r}: {path!r} must be a mapping, got {type(caps).__name__}"
            )
        unknown = set(caps) - {"allowed", "blocked"}
        if unknown:
            raise ValueError(
                f"policy {name!r}: {path!r} has unknown keys {sorted(unknown)}"
            )
        for key in ("allowed", "blocked"):
            lst = caps.get(key)
            if lst is not None:
                if not isinstance(lst, list):
                    raise ValueError(
                        f"policy {name!r}: {path}.{key} must be a list, "
                        f"got {type(lst).__name__}"
                    )
                for i, item in enumerate(lst):
                    if not isinstance(item, str):
                        raise ValueError(
                            f"policy {name!r}: {path}.{key}[{i}] must be a string, "
                            f"got {type(item).__name__}"
                        )

    def evaluate(
        self,
        application_id: str,
        text: str,
        capabilities: list[str],
        roles: list[str],
        pii_detector: PIIDetector,
    ) -> PolicyDecision:
        policy = self.policies.get(application_id)
        decision = PolicyDecision(evaluated=True)
        if policy is None:
            policy = STRICT_DEFAULT_POLICY
            decision.note = "no policy for application_id; strict default applied"

        # 1. Capabilities
        allowed, blocked = self._capability_lists(policy, roles)
        for cap in capabilities:
            if cap in blocked or (allowed is not None and cap not in allowed):
                decision.capability_violations.append(cap)
        if decision.capability_violations:
            decision.verdict = "block"
            decision.blocked_by = "capabilities"
            return decision

        # 2. Topics
        lowered = text.lower()
        for topic in (policy.get("topics", {}) or {}).get("blocked", []) or []:
            if any(kw.lower() in lowered for kw in topic.get("keywords", [])):
                decision.topic_matches.append(topic["id"])
        if decision.topic_matches:
            decision.verdict = "block"
            decision.blocked_by = "topics"
            return decision

        # 3. PII
        pii_policy = policy.get("pii") or {}
        action = pii_policy.get("action")
        if action:
            decision.pii_action = action
            entities = pii_detector.detect(text)
            if entities is None:
                # Fail safe: the policy wants PII handling but the model
                # can't run — hold the request for review.
                decision.verdict = "review"
                decision.note = (
                    f"pii detector unavailable: {pii_detector.load_error}"
                )
                return decision

            types_filter = pii_policy.get("entities")
            found = sorted(
                {e.type for e in entities
                 if types_filter is None or e.type in types_filter}
            )
            decision.pii_found = found
            if found:
                if action == "block":
                    decision.verdict = "block"
                    decision.blocked_by = "pii"
                elif action == "redact":
                    redacted, _ = redact(text, entities, types_filter)
                    decision.redacted_prompt = redacted
                # "flag": verdict unchanged, pii_found reports the finding

        return decision

    def evaluate_output(
        self,
        application_id: str,
        text: str,
        pii_detector: PIIDetector,
    ) -> PolicyDecision:
        """Policy checks that apply to a model response: topics and PII.

        Capabilities are a request-time concern (what the caller may ask
        the LLM to do) and are skipped here. Same policy file serves both
        directions; same downgrade-only verdict rules.
        """
        policy = self.policies.get(application_id)
        decision = PolicyDecision(evaluated=True)
        if policy is None:
            policy = STRICT_DEFAULT_POLICY
            decision.note = "no policy for application_id; strict default applied"

        lowered = text.lower()
        for topic in (policy.get("topics", {}) or {}).get("blocked", []) or []:
            if any(kw.lower() in lowered for kw in topic.get("keywords", [])):
                decision.topic_matches.append(topic["id"])
        if decision.topic_matches:
            decision.verdict = "block"
            decision.blocked_by = "topics"
            return decision

        pii_policy = policy.get("pii") or {}
        action = pii_policy.get("action")
        if action:
            decision.pii_action = action
            entities = pii_detector.detect(text)
            if entities is None:
                decision.verdict = "review"
                decision.note = (
                    f"pii detector unavailable: {pii_detector.load_error}"
                )
                return decision

            types_filter = pii_policy.get("entities")
            found = sorted(
                {e.type for e in entities
                 if types_filter is None or e.type in types_filter}
            )
            decision.pii_found = found
            if found:
                if action == "block":
                    decision.verdict = "block"
                    decision.blocked_by = "pii"
                elif action == "redact":
                    redacted, _ = redact(text, entities, types_filter)
                    decision.redacted_prompt = redacted

        return decision

    @staticmethod
    def _capability_lists(
        policy: dict, roles: list[str]
    ) -> tuple[list[str] | None, list[str]]:
        """Return (allowed, blocked); role-scoped lists override defaults.

        allowed=None means the policy declares no allowlist, so anything
        not explicitly blocked passes.
        """
        caps = policy.get("capabilities", {}) or {}
        for role in roles:
            role_caps = (
                ((policy.get("roles", {}) or {}).get(role, {}) or {})
                .get("capabilities")
            )
            if role_caps is not None:
                caps = role_caps
                break
        return caps.get("allowed"), caps.get("blocked") or []
