"""Request schema — pipeline stage 1 (request_validation).

FastAPI + Pydantic reject malformed requests with a 422 before any
pipeline logic runs. Limits and patterns are read from app.config.
"""
import re

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.config import settings

# Compiled once at import time from the config value.
_CONTROL_CHARS = re.compile(f"[{settings.control_chars_pattern}]")


class PromptContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    application_id: str | None = Field(default=None, max_length=128)
    user_id: str | None = Field(default=None, max_length=128)
    session_id: str | None = Field(default=None, max_length=128)
    # Capabilities the calling app intends to grant the LLM for this
    # request; checked against the application's policy.
    capabilities: list[str] = Field(default_factory=list, max_length=32)


class Thresholds(BaseModel):
    model_config = ConfigDict(extra="forbid")

    medium: float = Field(default=settings.classifier_threshold_medium, ge=0.0, le=1.0)
    high: float = Field(default=settings.classifier_threshold_high, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def high_above_medium(self) -> "Thresholds":
        if self.high <= self.medium:
            raise ValueError("thresholds.high must be greater than thresholds.medium")
        return self


class AnalyzeOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    thresholds: Thresholds = Field(default_factory=Thresholds)


class AnalyzeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt: str = Field(min_length=1, max_length=settings.max_prompt_length)
    context: PromptContext | None = None
    options: AnalyzeOptions = Field(default_factory=AnalyzeOptions)

    @field_validator("prompt")
    @classmethod
    def no_control_chars(cls, v: str) -> str:
        if _CONTROL_CHARS.search(v):
            raise ValueError("prompt contains disallowed control characters")
        return v


class OutputAnalyzeRequest(BaseModel):
    """Request schema for POST /v1/output/analyze.

    `output` is the model response to vet. `prompt` is the user prompt it
    answered — optional, but recommended: Llama Guard judges the response
    in the context of the conversation, so supplying it improves accuracy.
    """

    model_config = ConfigDict(extra="forbid")

    output: str = Field(min_length=1, max_length=settings.max_output_length)
    prompt: str | None = Field(default=None, max_length=settings.max_prompt_length)
    context: PromptContext | None = None

    @field_validator("output", "prompt")
    @classmethod
    def no_control_chars(cls, v: str | None) -> str | None:
        if v is not None and _CONTROL_CHARS.search(v):
            raise ValueError("field contains disallowed control characters")
        return v
