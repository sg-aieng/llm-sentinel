"""Response schema for POST /v1/input/analyze."""
from typing import Literal

from pydantic import BaseModel

Verdict = Literal["allow", "review", "block"]


class SchemaStageResult(BaseModel):
    passed: bool = True


class NormalizationStageResult(BaseModel):
    applied: list[str] = []


class RuleChecksStageResult(BaseModel):
    passed: bool
    violations: list[str] = []


class ClassifierStageResult(BaseModel):
    ran: bool
    score: float | None = None
    model: str | None = None
    note: str | None = None


class PIIPolicyResult(BaseModel):
    found: list[str] = []
    action: str | None = None


class PolicyStageResult(BaseModel):
    evaluated: bool
    capability_violations: list[str] = []
    topic_matches: list[str] = []
    pii: PIIPolicyResult | None = None
    note: str | None = None


class PipelineDetail(BaseModel):
    schema_validation: SchemaStageResult = SchemaStageResult()
    normalization: NormalizationStageResult = NormalizationStageResult()
    rule_checks: RuleChecksStageResult | None = None
    classifier: ClassifierStageResult | None = None
    policy: PolicyStageResult | None = None


class ResponseMeta(BaseModel):
    latency_ms: int
    timestamp: str


class AnalyzeResponse(BaseModel):
    request_id: str
    verdict: Verdict
    risk_score: float | None = None
    blocked_at: Literal["rule_checks", "classifier", "policy"] | None = None
    sanitized_prompt: str | None = None
    prompt_hash: str
    pipeline: PipelineDetail
    meta: ResponseMeta


# --- POST /v1/output/analyze ---


class SafetyStageResult(BaseModel):
    """Llama Guard verdict: binary safe/unsafe plus hazard categories —
    no score, unlike the input classifier."""

    ran: bool
    safe: bool | None = None
    categories: list[str] = []
    model: str | None = None
    note: str | None = None


class OutputPipelineDetail(BaseModel):
    # Field order mirrors execution order: policy (redaction) runs before
    # the safety classifier so Llama Guard judges the sanitized text.
    schema_validation: SchemaStageResult = SchemaStageResult()
    normalization: NormalizationStageResult = NormalizationStageResult()
    rule_checks: RuleChecksStageResult | None = None
    policy: PolicyStageResult | None = None
    safety: SafetyStageResult | None = None


class OutputAnalyzeResponse(BaseModel):
    request_id: str
    verdict: Verdict
    blocked_at: Literal["rule_checks", "safety", "policy"] | None = None
    safety_categories: list[str] = []
    sanitized_output: str | None = None
    output_hash: str
    pipeline: OutputPipelineDetail
    meta: ResponseMeta
