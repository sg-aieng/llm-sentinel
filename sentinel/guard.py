"""llm-sentinel Python SDK — in-process guard with no server required.

Drop the REST API. Import Guard, call check_input / check_output, done::

    from sentinel import Guard, PromptBlocked

    guard = Guard.from_policy("policies")

    result = guard.check_input(prompt, user_id="u123")
    if result.verdict == "block":
        raise PromptBlocked(result)

    llm_response = my_llm(result.sanitized_prompt)

    out = guard.check_output(llm_response, prompt=prompt)
    if out.verdict != "block":
        return out.sanitized_output
"""
import re
import threading
import time
import uuid
from pathlib import Path

from app.audit import audit_log, prompt_hash
from app.config import settings

# Compiled once at import time — mirrors app/models/request.py.
_CONTROL_CHARS = re.compile(f"[{settings.control_chars_pattern}]")


def _validate_prompt_text(prompt: str) -> None:
    if not prompt:
        raise ValueError("prompt must not be empty")
    if len(prompt) > settings.max_prompt_length:
        raise ValueError(f"prompt exceeds maximum length of {settings.max_prompt_length}")
    if _CONTROL_CHARS.search(prompt):
        raise ValueError("prompt contains disallowed control characters")


def _validate_output_text(output: str) -> None:
    if not output:
        raise ValueError("output must not be empty")
    if len(output) > settings.max_output_length:
        raise ValueError(f"output exceeds maximum length of {settings.max_output_length}")
    if _CONTROL_CHARS.search(output):
        raise ValueError("output contains disallowed control characters")


def _validate_thresholds(medium: float, high: float) -> None:
    if not 0.0 <= medium <= 1.0:
        raise ValueError("threshold_medium must be between 0.0 and 1.0")
    if not 0.0 <= high <= 1.0:
        raise ValueError("threshold_high must be between 0.0 and 1.0")
    if high <= medium:
        raise ValueError("threshold_high must be greater than threshold_medium")
from app.identity import StaticIdentityProvider
from app.models.response import (
    AnalyzeResponse,
    ClassifierStageResult,
    NormalizationStageResult,
    OutputAnalyzeResponse,
    OutputPipelineDetail,
    PIIPolicyResult,
    PipelineDetail,
    PolicyStageResult,
    ResponseMeta,
    RuleChecksStageResult,
    SafetyStageResult,
)
from app.pipeline.classifier import PromptGuardClassifier
from app.pipeline.normalization import normalize
from app.pipeline.pii import PIIDetector
from app.pipeline.policy import PolicyEngine
from app.pipeline.rules import RuleEngine
from app.pipeline.safety import LlamaGuardClassifier


class PromptBlocked(Exception):
    """Raised by check_input when raise_on_block=True and verdict is block."""

    def __init__(self, result: AnalyzeResponse) -> None:
        self.result = result
        self.blocked_at = result.blocked_at
        self.violations: list[str] = (
            result.pipeline.rule_checks.violations
            if result.pipeline.rule_checks
            else []
        )
        super().__init__(f"prompt blocked at {self.blocked_at}: {self.violations}")


class OutputBlocked(Exception):
    """Raised by check_output when raise_on_block=True and verdict is block."""

    def __init__(self, result: OutputAnalyzeResponse) -> None:
        self.result = result
        self.blocked_at = result.blocked_at
        self.safety_categories: list[str] = result.safety_categories or []
        super().__init__(
            f"output blocked at {self.blocked_at}: {self.safety_categories}"
        )


class Guard:
    """In-process llm-sentinel guard. No server. No network hop.

    Models load lazily on the first check_input / check_output call,
    or eagerly via guard.load().

    Args:
        policy_dir: Directory of per-application policy YAML files.
            Defaults to the ``policies/`` folder configured in settings.
        identity_provider: Custom IAM implementation of the
            ``IdentityProvider`` protocol. Defaults to
            ``StaticIdentityProvider`` (reads identities.yaml).
    """

    def __init__(
        self,
        *,
        policy_dir: str | Path | None = None,
        identity_provider=None,
    ) -> None:
        self._policy_dir = Path(policy_dir) if policy_dir else None
        self._identity_provider_override = identity_provider
        self._load_lock = threading.Lock()
        self._loaded = False

        self._rule_engine: RuleEngine | None = None
        self._output_rule_engine: RuleEngine | None = None
        self._classifier: PromptGuardClassifier | None = None
        self._safety: LlamaGuardClassifier | None = None
        self._policy_engine: PolicyEngine | None = None
        self._pii: PIIDetector | None = None
        self._identity_provider = None

    @classmethod
    def from_policy(cls, policy_dir: str | Path) -> "Guard":
        """Create a Guard scoped to a specific policy directory.

        Args:
            policy_dir: Directory containing per-application policy YAML files.
                Pass the **directory** itself, not a file inside it.
                Example: ``Guard.from_policy("policies")``

        Raises:
            ValueError: if ``policy_dir`` is an existing file (not a directory).
        """
        path = Path(policy_dir)
        if path.exists() and not path.is_dir():
            raise ValueError(
                f"policy_dir must be a directory, not a file: {path!r}. "
                "Pass the directory containing policy YAML files, e.g. 'policies'."
            )
        return cls(policy_dir=path)

    def load(self) -> "Guard":
        """Eagerly load all models. Returns self for chaining."""
        self._ensure_loaded()
        return self

    # ------------------------------------------------------------------
    # Input guard
    # ------------------------------------------------------------------

    def check_input(
        self,
        prompt: str,
        *,
        application_id: str | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
        capabilities: list[str] | None = None,
        threshold_medium: float | None = None,
        threshold_high: float | None = None,
        raise_on_block: bool = False,
    ) -> AnalyzeResponse:
        """Run the input guard pipeline and return the full result.

        Args:
            prompt: The user prompt to analyze.
            application_id: Enables per-application policy evaluation.
            user_id: Resolved via the identity provider for role-scoped policies.
            capabilities: Capabilities the user is requesting (e.g. ``["web_search"]``).
            threshold_medium: Classifier score at which verdict becomes ``review``.
            threshold_high: Classifier score at which verdict becomes ``block``.
            raise_on_block: Raise ``PromptBlocked`` instead of returning the result.

        Returns:
            ``AnalyzeResponse`` with ``verdict``, ``sanitized_prompt``, and
            full ``pipeline`` detail.
        """
        _validate_prompt_text(prompt)
        t_medium = threshold_medium if threshold_medium is not None else settings.classifier_threshold_medium
        t_high = threshold_high if threshold_high is not None else settings.classifier_threshold_high
        _validate_thresholds(t_medium, t_high)

        self._ensure_loaded()
        start = time.perf_counter()
        request_id = str(uuid.uuid4())
        hashed = prompt_hash(prompt)
        pipeline = PipelineDetail()

        norm = normalize(prompt)
        pipeline.normalization = NormalizationStageResult(applied=norm.applied)

        rule_result = self._rule_engine.check(norm.text)
        pipeline.rule_checks = RuleChecksStageResult(
            passed=rule_result.passed, violations=rule_result.violations
        )
        if not rule_result.passed:
            audit_log(
                request_id, "rule_checks", "block", hashed,
                application_id=application_id, user_id=user_id,
                session_id=session_id, violations=rule_result.violations,
            )
            result = _input_response(
                request_id, "block", pipeline, hashed, start, blocked_at="rule_checks"
            )
            if raise_on_block:
                raise PromptBlocked(result)
            return result

        scored = self._classifier.score(norm.text)
        pipeline.classifier = ClassifierStageResult(
            ran=scored.ran, score=scored.score, model=scored.model, note=scored.note
        )

        if not scored.ran:
            verdict, risk_score = "review", None
        elif scored.score >= t_high:
            verdict, risk_score = "block", scored.score
        elif scored.score >= t_medium:
            verdict, risk_score = "review", scored.score
        else:
            verdict, risk_score = "allow", scored.score

        audit_log(
            request_id, "classifier", verdict, hashed,
            application_id=application_id, user_id=user_id,
            session_id=session_id, risk_score=risk_score,
        )
        if verdict == "block":
            result = _input_response(
                request_id, verdict, pipeline, hashed, start,
                risk_score=risk_score, blocked_at="classifier",
            )
            if raise_on_block:
                raise PromptBlocked(result)
            return result

        sanitized = norm.text
        if application_id is not None:
            roles = ["default"]
            if user_id:
                identity = self._identity_provider.resolve(user_id)
                roles = identity.roles
            decision = self._policy_engine.evaluate(
                application_id=application_id,
                text=norm.text,
                capabilities=capabilities or [],
                roles=roles,
                pii_detector=self._pii,
            )
            pipeline.policy = PolicyStageResult(
                evaluated=True,
                capability_violations=decision.capability_violations,
                topic_matches=decision.topic_matches,
                pii=PIIPolicyResult(found=decision.pii_found, action=decision.pii_action)
                if decision.pii_action
                else None,
                note=decision.note,
            )
            if decision.redacted_prompt is not None:
                sanitized = decision.redacted_prompt
            if decision.verdict == "block":
                audit_log(
                    request_id, "policy", "block", hashed,
                    application_id=application_id, user_id=user_id,
                    session_id=session_id,
                    violations=(
                        decision.capability_violations
                        + decision.topic_matches
                        + ([f"pii:{t}" for t in decision.pii_found] if decision.blocked_by == "pii" else [])
                    ),
                )
                result = _input_response(
                    request_id, "block", pipeline, hashed, start,
                    risk_score=risk_score, blocked_at="policy",
                )
                if raise_on_block:
                    raise PromptBlocked(result)
                return result
            if decision.verdict == "review":
                verdict = "review"
                # PII handling was configured but the detector couldn't run —
                # don't hand back potentially PII-containing text as sanitized.
                if decision.note and "unavailable" in decision.note:
                    sanitized = None
            audit_log(
                request_id, "policy", verdict, hashed,
                application_id=application_id, user_id=user_id, session_id=session_id,
            )

        return _input_response(
            request_id, verdict, pipeline, hashed, start,
            risk_score=risk_score, sanitized_prompt=sanitized,
        )

    # ------------------------------------------------------------------
    # Output guard
    # ------------------------------------------------------------------

    def check_output(
        self,
        output: str,
        *,
        prompt: str | None = None,
        application_id: str | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
        raise_on_block: bool = False,
    ) -> OutputAnalyzeResponse:
        """Run the output guard pipeline and return the full result.

        Args:
            output: The model response to vet.
            prompt: The original user prompt (improves Llama Guard accuracy).
            application_id: Enables per-application policy evaluation.
            raise_on_block: Raise ``OutputBlocked`` instead of returning the result.

        Returns:
            ``OutputAnalyzeResponse`` with ``verdict``, ``sanitized_output``,
            ``safety_categories``, and full ``pipeline`` detail.
        """
        _validate_output_text(output)

        self._ensure_loaded()
        start = time.perf_counter()
        request_id = str(uuid.uuid4())
        hashed = prompt_hash(output)
        pipeline = OutputPipelineDetail()

        norm = normalize(output)
        pipeline.normalization = NormalizationStageResult(applied=norm.applied)

        rule_result = self._output_rule_engine.check(norm.text)
        pipeline.rule_checks = RuleChecksStageResult(
            passed=rule_result.passed, violations=rule_result.violations
        )
        if not rule_result.passed:
            audit_log(
                request_id, "output_rule_checks", "block", hashed,
                application_id=application_id, user_id=user_id,
                session_id=session_id, violations=rule_result.violations,
            )
            result = _output_response(
                request_id, "block", pipeline, hashed, start, blocked_at="rule_checks"
            )
            if raise_on_block:
                raise OutputBlocked(result)
            return result

        verdict = "allow"
        sanitized = norm.text
        pii_handled = False
        if application_id is not None:
            decision = self._policy_engine.evaluate_output(
                application_id=application_id,
                text=norm.text,
                pii_detector=self._pii,
            )
            pipeline.policy = PolicyStageResult(
                evaluated=True,
                topic_matches=decision.topic_matches,
                pii=PIIPolicyResult(found=decision.pii_found, action=decision.pii_action)
                if decision.pii_action
                else None,
                note=decision.note,
            )
            pii_detector_failed = (
                decision.note is not None and "unavailable" in decision.note
            )
            # Only tell Llama Guard to skip S7 (privacy) when PII detection
            # actually ran — not when it was attempted but unavailable.
            pii_handled = decision.pii_action is not None and not pii_detector_failed
            if decision.redacted_prompt is not None:
                sanitized = decision.redacted_prompt
            if decision.verdict == "block":
                audit_log(
                    request_id, "output_policy", "block", hashed,
                    application_id=application_id, user_id=user_id,
                    session_id=session_id,
                    violations=(
                        decision.topic_matches
                        + ([f"pii:{t}" for t in decision.pii_found] if decision.blocked_by == "pii" else [])
                    ),
                )
                result = _output_response(
                    request_id, "block", pipeline, hashed, start, blocked_at="policy"
                )
                if raise_on_block:
                    raise OutputBlocked(result)
                return result
            if decision.verdict == "review":
                verdict = "review"
                # PII handling was configured but the detector couldn't run —
                # don't hand back potentially PII-containing text as sanitized.
                if pii_detector_failed:
                    sanitized = None
            audit_log(
                request_id, "output_policy", verdict, hashed,
                application_id=application_id, user_id=user_id, session_id=session_id,
            )

        safety_result = self._safety.classify(
            sanitized,
            prompt=prompt,
            extra_excluded_categories=["S7"] if pii_handled else None,
        )
        pipeline.safety = SafetyStageResult(
            ran=safety_result.ran, safe=safety_result.safe,
            categories=safety_result.categories,
            model=safety_result.model, note=safety_result.note,
        )
        if safety_result.ran and safety_result.safe is False:
            audit_log(
                request_id, "safety", "block", hashed,
                application_id=application_id, user_id=user_id,
                session_id=session_id, violations=safety_result.categories,
            )
            result = _output_response(
                request_id, "block", pipeline, hashed, start,
                blocked_at="safety", safety_categories=safety_result.categories,
            )
            if raise_on_block:
                raise OutputBlocked(result)
            return result
        if not safety_result.ran:
            verdict = "review"
        audit_log(
            request_id, "safety", verdict, hashed,
            application_id=application_id, user_id=user_id, session_id=session_id,
        )

        return _output_response(
            request_id, verdict, pipeline, hashed, start, sanitized_output=sanitized
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        with self._load_lock:
            if self._loaded:
                return
            from importlib.resources import files as _rf
            self._rule_engine = RuleEngine()
            self._output_rule_engine = RuleEngine(
                rules_path=Path(str(_rf("app") / "rules" / settings.output_rules_file))
            )
            self._classifier = PromptGuardClassifier()
            self._classifier.load()
            self._safety = LlamaGuardClassifier()
            self._safety.load()
            self._policy_engine = PolicyEngine(
                policies_dir=self._policy_dir if self._policy_dir else None
            )
            self._pii = PIIDetector()
            self._pii.load()
            self._identity_provider = self._identity_provider_override or StaticIdentityProvider()
            self._loaded = True


# ------------------------------------------------------------------
# Private response builders (mirrors the FastAPI _respond helpers)
# ------------------------------------------------------------------

def _utc_now() -> str:
    from app.audit import utc_now
    return utc_now()


def _input_response(
    request_id: str,
    verdict: str,
    pipeline: PipelineDetail,
    hashed: str,
    start: float,
    *,
    risk_score: float | None = None,
    blocked_at: str | None = None,
    sanitized_prompt: str | None = None,
) -> AnalyzeResponse:
    return AnalyzeResponse(
        request_id=request_id,
        verdict=verdict,
        risk_score=risk_score,
        blocked_at=blocked_at,
        sanitized_prompt=sanitized_prompt,
        prompt_hash=hashed,
        pipeline=pipeline,
        meta=ResponseMeta(
            latency_ms=int((time.perf_counter() - start) * 1000),
            timestamp=_utc_now(),
        ),
    )


def _output_response(
    request_id: str,
    verdict: str,
    pipeline: OutputPipelineDetail,
    hashed: str,
    start: float,
    *,
    blocked_at: str | None = None,
    safety_categories: list[str] | None = None,
    sanitized_output: str | None = None,
) -> OutputAnalyzeResponse:
    return OutputAnalyzeResponse(
        request_id=request_id,
        verdict=verdict,
        blocked_at=blocked_at,
        safety_categories=safety_categories or [],
        sanitized_output=sanitized_output,
        output_hash=hashed,
        pipeline=pipeline,
        meta=ResponseMeta(
            latency_ms=int((time.perf_counter() - start) * 1000),
            timestamp=_utc_now(),
        ),
    )
