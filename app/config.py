"""Application configuration.

Values are read in this order (highest wins):
  1. Environment variables prefixed with SENTINEL_
  2. .env file in the project root
  3. Defaults defined here

Typical .env file:

    SENTINEL_MAX_PROMPT_LENGTH=4096
    SENTINEL_CONTROL_CHARS_PATTERN=[\\x00-\\x08\\x0b\\x0c\\x0e-\\x1f\\x7f]
    SENTINEL_CLASSIFIER_MODEL=meta-llama/Llama-Prompt-Guard-2-86M
    SENTINEL_DISABLE_CLASSIFIER=false
    SENTINEL_CLASSIFIER_THRESHOLD_MEDIUM=0.50
    SENTINEL_CLASSIFIER_THRESHOLD_HIGH=0.90
"""
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_ENV_FILE = Path(__file__).resolve().parents[1] / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="SENTINEL_",
        env_file=_ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- request validation ---
    max_prompt_length: int = 8192
    # Raw regex character class body (without surrounding brackets).
    # Default blocks control chars below 0x20 except \n (0x0a) and \t (0x09),
    # plus DEL (0x7f).
    control_chars_pattern: str = r"\x00-\x08\x0b\x0c\x0e-\x1f\x7f"

    # --- normalization ---
    # Full regex pattern for invisible / zero-width characters to strip.
    # Covers U+200B-U+200F (zero-width/LRM/RLM), U+2028-U+2029 (line/para
    # separators), U+202A-U+202E (bidi overrides), U+2060-U+2064 (word
    # joiner / invisible operators), U+FEFF (BOM).
    invisible_chars_pattern: str = (
        "\u200b-\u200f\u2028\u2029\u202a-\u202e\u2060-\u2064\ufeff"
    )
    # Minimum consecutive spaces/tabs before collapsing to one space.
    max_space_run: int = 2
    # Minimum consecutive newlines before collapsing to two newlines.
    max_newline_run: int = 3

    # --- rule checks ---
    # Filename of the YAML rule pack inside the app/rules/ package directory.
    rules_file: str = "default.yaml"
    # Regex for base64-like blobs (full pattern, not a character class body).
    base64_blob_pattern: str = r"[A-Za-z0-9+/=]{40,}"
    # Regex for hex-encoded blobs.
    hex_blob_pattern: str = r"(?:[0-9a-fA-F]{2}){20,}"
    # Fraction of prompt that may be base64/hex before flagging.
    max_encoded_ratio: float = 0.4
    # Minimum blob length (chars) before it counts toward the ratio.
    min_blob_length: int = 40

    # --- classifier ---
    classifier_model: str = "meta-llama/Llama-Prompt-Guard-2-86M"
    disable_classifier: bool = False
    classifier_threshold_medium: float = 0.50
    classifier_threshold_high: float = 0.90
    # Device for classifier: auto (cuda→mps→cpu), or explicit (cuda/mps/cpu).
    classifier_device: str = "auto"

    # --- output guard ---
    # Maximum model-output length in characters.
    max_output_length: int = 16384
    # Filename of the output rule pack inside the app/rules/ package directory.
    output_rules_file: str = "output.yaml"
    # Llama Guard content-safety model. Generates safe/unsafe verdicts with
    # hazard categories (S1..S14). Gated on Hugging Face — request access
    # or point at an ungated mirror.
    safety_model: str = "meta-llama/Llama-Guard-3-1B"
    disable_safety: bool = False
    # Verdict + category codes fit comfortably in a few tokens.
    safety_max_new_tokens: int = 20
    # Device for safety classifier: auto (cuda→mps→cpu), or explicit (cuda/mps/cpu).
    safety_device: str = "auto"
    # Comma-separated Llama Guard hazard codes to exclude from enforcement,
    # e.g. "S6,S13" to allow specialized advice and elections content.
    safety_excluded_categories: str = ""

    # --- REST API authentication ---
    # When set, every request to the REST API must include the header:
    #   Authorization: Bearer <value>
    # Leave unset (empty) only in trusted network environments where the API
    # is not reachable from untrusted callers (mTLS, VPC policy, service mesh).
    api_key: str = ""

    # --- policy ---
    # Directory of per-application policy YAML files (<application_id>.yaml).
    policies_dir: str = "policies"
    # PII token-classification model for the redact/block/flag actions.
    pii_model: str = "iiiorg/piiranha-v1-detect-personal-information"
    # Secondary NER model: piiranha is strong on structured PII (emails,
    # phones, ids) but misses person names in prose; this fills that gap.
    # Set empty to disable the second model.
    pii_ner_model: str = "dslim/bert-base-NER"
    disable_pii: bool = False
    # Optional static identity file for the placeholder IdentityProvider.
    identities_file: str = "identities.yaml"

    @field_validator("max_prompt_length")
    @classmethod
    def positive_length(cls, v: int) -> int:
        if v < 1:
            raise ValueError("max_prompt_length must be >= 1")
        return v

    @field_validator("classifier_threshold_high")
    @classmethod
    def high_above_medium(cls, v: float, info) -> float:
        medium = (info.data or {}).get("classifier_threshold_medium", 0.50)
        if v <= medium:
            raise ValueError("classifier_threshold_high must be > classifier_threshold_medium")
        return v


settings = Settings()
