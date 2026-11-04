"""POST /v1/input/analyze — the input guard pipeline."""
import time
import uuid

from fastapi import APIRouter, Request

from app.audit import audit_log, prompt_hash, utc_now
from app.models.request import AnalyzeRequest
from app.models.response import (
    AnalyzeResponse,
    ClassifierStageResult,
    NormalizationStageResult,
    PIIPolicyResult,
    PipelineDetail,
    PolicyStageResult,
    ResponseMeta,
    RuleChecksStageResult,
)
from app.pipeline.normalization import normalize

router = APIRouter()


@router.post("/input/analyze", response_model=AnalyzeResponse, response_model_exclude_none=True)
def analyze_input(body: AnalyzeRequest, request: Request) -> AnalyzeResponse:
    start = time.perf_counter()
    request_id = str(uuid.uuid4())
    app_id = body.context.application_id if body.context else None
    user_id = body.context.user_id if body.context else None
    session_id = body.context.session_id if body.context else None
    hashed = prompt_hash(body.prompt)

    pipeline = PipelineDetail()

    # Stage 2 — normalization (stage 1, request validation, already ran via Pydantic)
    norm = normalize(body.prompt)
    pipeline.normalization = NormalizationStageResult(applied=norm.applied)

    # Stage 3 — rule checks
    rule_engine = request.app.state.rule_engine
    rule_result = rule_engine.check(norm.text)
    pipeline.rule_checks = RuleChecksStageResult(
        passed=rule_result.passed, violations=rule_result.violations
    )

    if not rule_result.passed:
        audit_log(
            request_id, "rule_checks", "block", hashed,
            application_id=app_id, user_id=user_id, session_id=session_id,
            violations=rule_result.violations,
        )
        return _respond(
            request_id, "block", pipeline, hashed, start,
            blocked_at="rule_checks",
        )

    # Stage 4 — classifier
    classifier = request.app.state.classifier
    result = classifier.score(norm.text)
    pipeline.classifier = ClassifierStageResult(
        ran=result.ran, score=result.score, model=result.model, note=result.note
    )

    thresholds = body.options.thresholds
    if not result.ran:
        # Fail safe: without a classifier verdict we don't allow outright.
        verdict = "review"
        risk_score = None
    elif result.score >= thresholds.high:
        verdict = "block"
        risk_score = result.score
    elif result.score >= thresholds.medium:
        verdict = "review"
        risk_score = result.score
    else:
        verdict = "allow"
        risk_score = result.score

    audit_log(
        request_id, "classifier", verdict, hashed,
        application_id=app_id, user_id=user_id, session_id=session_id,
        risk_score=risk_score,
    )
    if verdict == "block":
        return _respond(
            request_id, verdict, pipeline, hashed, start,
            risk_score=risk_score, blocked_at="classifier",
        )

    # Stage 5 — policy (only when the caller identifies an application)
    # TRUST MODEL: application_id and user_id come from the calling service, not
    # the end user. The REST API is a trusted internal sidecar — secure it at the
    # network layer (mTLS, VPC policy, or a service mesh) and never expose it
    # directly to external clients.
    sanitized = norm.text
    if app_id is not None:
        roles = ["default"]
        if body.context.user_id:
            identity = request.app.state.identity_provider.resolve(body.context.user_id)
            roles = identity.roles
        decision = request.app.state.policy_engine.evaluate(
            application_id=app_id,
            text=norm.text,
            capabilities=body.context.capabilities,
            roles=roles,
            pii_detector=request.app.state.pii_detector,
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
                application_id=app_id, user_id=user_id, session_id=session_id,
                violations=decision.capability_violations + decision.topic_matches
                + ([f"pii:{t}" for t in decision.pii_found] if decision.blocked_by == "pii" else []),
            )
            return _respond(
                request_id, "block", pipeline, hashed, start,
                risk_score=risk_score, blocked_at="policy",
            )
        if decision.verdict == "review":
            verdict = "review"
        audit_log(
            request_id, "policy", verdict, hashed,
            application_id=app_id, user_id=user_id, session_id=session_id,
        )

    return _respond(
        request_id, verdict, pipeline, hashed, start,
        risk_score=risk_score,
        sanitized_prompt=sanitized,
    )


def _respond(
    request_id: str,
    verdict: str,
    pipeline: PipelineDetail,
    hashed: str,
    start: float,
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
            timestamp=utc_now(),
        ),
    )
