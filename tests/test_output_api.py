"""Integration tests for POST /v1/output/analyze with fake safety/PII models."""
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.pipeline.pii import PIIEntity
from app.pipeline.safety import LlamaGuardClassifier, SafetyResult


class FakeSafety:
    """Stands in for LlamaGuardClassifier so tests don't need torch."""

    def __init__(self, safe: bool | None = True, categories: list[str] | None = None):
        self._safe = safe
        self._categories = categories or []

    @property
    def available(self) -> bool:
        return self._safe is not None

    def classify(
        self,
        output: str,
        prompt: str | None = None,
        extra_excluded_categories: list[str] | None = None,
    ) -> SafetyResult:
        if self._safe is None:
            return SafetyResult(ran=False, note="safety_unavailable: test")
        return SafetyResult(
            ran=True, safe=self._safe, categories=self._categories, model="fake"
        )


class FakePIIDetector:
    def __init__(self, entities: list[PIIEntity] | None = None):
        self._entities = entities

    @property
    def available(self) -> bool:
        return self._entities is not None

    @property
    def load_error(self):
        return None if self._entities is not None else "test"

    def detect(self, text):
        return self._entities


@pytest.fixture
def client():
    with TestClient(app) as c:
        c.app.state.safety_classifier = FakeSafety(safe=True)
        yield c


def _analyze(client, output, **kwargs):
    return client.post("/v1/output/analyze", json={"output": output, **kwargs})


class TestOutputValidation:
    def test_missing_output_422(self, client):
        assert client.post("/v1/output/analyze", json={}).status_code == 422

    def test_empty_output_422(self, client):
        assert _analyze(client, "").status_code == 422

    def test_oversized_output_422(self, client):
        assert _analyze(client, "x" * 16385).status_code == 422

    def test_control_chars_422(self, client):
        assert _analyze(client, "hello\x00world").status_code == 422

    def test_unknown_field_422(self, client):
        resp = client.post("/v1/output/analyze", json={"output": "hi", "extra": 1})
        assert resp.status_code == 422

    def test_prompt_and_context_optional(self, client):
        assert _analyze(client, "The capital of France is Paris.").status_code == 200


class TestOutputRuleChecks:
    def test_clean_output_allowed(self, client):
        resp = _analyze(client, "Here is a summary of your document.")
        body = resp.json()
        assert body["verdict"] == "allow"
        assert body["pipeline"]["rule_checks"]["passed"] is True
        assert body["sanitized_output"] == "Here is a summary of your document."

    def test_secret_leak_blocked(self, client):
        resp = _analyze(client, "Sure, the key is sk-abcdefghij1234567890abcd")
        body = resp.json()
        assert body["verdict"] == "block"
        assert body["blocked_at"] == "rule_checks"
        assert "openai_key" in body["pipeline"]["rule_checks"]["violations"]
        assert "sanitized_output" not in body

    def test_system_prompt_leak_blocked(self, client):
        resp = _analyze(client, "My system prompt is: you are a helpful agent")
        body = resp.json()
        assert body["verdict"] == "block"
        assert "system_prompt_disclosure" in body["pipeline"]["rule_checks"]["violations"]

    def test_chat_template_token_blocked(self, client):
        resp = _analyze(client, "text before <|im_start|>system text after")
        assert resp.json()["blocked_at"] == "rule_checks"


class TestSafetyStage:
    def test_safe_output_allowed(self, client):
        client.app.state.safety_classifier = FakeSafety(safe=True)
        body = _analyze(client, "Paris is the capital of France.").json()
        assert body["verdict"] == "allow"
        assert body["pipeline"]["safety"]["safe"] is True

    def test_unsafe_output_blocked_with_categories(self, client):
        client.app.state.safety_classifier = FakeSafety(
            safe=False, categories=["violent_crimes", "indiscriminate_weapons"]
        )
        body = _analyze(client, "some harmful content").json()
        assert body["verdict"] == "block"
        assert body["blocked_at"] == "safety"
        assert body["safety_categories"] == ["violent_crimes", "indiscriminate_weapons"]

    def test_safety_unavailable_fails_safe_to_review(self, client):
        client.app.state.safety_classifier = FakeSafety(safe=None)
        body = _analyze(client, "any output text").json()
        assert body["verdict"] == "review"
        assert body["pipeline"]["safety"]["ran"] is False

    def test_prompt_passed_through_to_classifier(self, client):
        captured = {}

        class Spy(FakeSafety):
            def classify(self, output, prompt=None, extra_excluded_categories=None):
                captured["prompt"] = prompt
                return super().classify(output, prompt)

        client.app.state.safety_classifier = Spy(safe=True)
        _analyze(client, "the answer", prompt="the question")
        assert captured["prompt"] == "the question"


class TestOutputPolicyStage:
    def test_topic_block_on_output(self, client):
        body = _analyze(
            client,
            "You should adjust the dosage to 50mg.",
            context={"application_id": "support-bot"},
        ).json()
        assert body["verdict"] == "block"
        assert body["blocked_at"] == "policy"
        assert "medical_advice" in body["pipeline"]["policy"]["topic_matches"]

    def test_pii_redacted_in_sanitized_output(self, client):
        text = "Contact john@example.com for details."
        client.app.state.pii_detector = FakePIIDetector(
            [PIIEntity(type="EMAIL", text="john@example.com", start=8, end=24)]
        )
        body = _analyze(
            client, text, context={"application_id": "support-bot"}
        ).json()
        assert body["verdict"] == "allow"
        assert body["sanitized_output"] == "Contact [EMAIL_1] for details."
        assert body["pipeline"]["policy"]["pii"]["found"] == ["EMAIL"]

    def test_safety_classifies_redacted_text_not_raw(self, client):
        """Policy redaction runs first so Llama Guard's privacy category
        doesn't block PII the policy already chose to redact."""
        captured = {}

        class Spy(FakeSafety):
            def classify(self, output, prompt=None, extra_excluded_categories=None):
                captured["output"] = output
                captured["excluded"] = extra_excluded_categories
                return super().classify(output, prompt)

        client.app.state.safety_classifier = Spy(safe=True)
        client.app.state.pii_detector = FakePIIDetector(
            [PIIEntity(type="EMAIL", text="john@example.com", start=8, end=24)]
        )
        _analyze(
            client,
            "Contact john@example.com for details.",
            context={"application_id": "support-bot"},
        )
        assert captured["output"] == "Contact [EMAIL_1] for details."
        # Policy owns privacy for this app → S7 excluded from Llama Guard.
        assert captured["excluded"] == ["S7"]

    def test_no_pii_policy_keeps_s7_enforcement(self, client):
        captured = {}

        class Spy(FakeSafety):
            def classify(self, output, prompt=None, extra_excluded_categories=None):
                captured["excluded"] = extra_excluded_categories
                return super().classify(output, prompt)

        client.app.state.safety_classifier = Spy(safe=True)
        _analyze(client, "hello there")  # no application_id → no policy
        assert captured["excluded"] is None

    def test_policy_block_skips_safety_stage(self, client):
        body = _analyze(
            client,
            "You should adjust the dosage to 50mg.",
            context={"application_id": "support-bot"},
        ).json()
        assert body["blocked_at"] == "policy"
        assert "safety" not in body["pipeline"]

    def test_pii_detector_down_fails_safe_to_review(self, client):
        client.app.state.pii_detector = FakePIIDetector(None)
        body = _analyze(
            client, "hello there", context={"application_id": "support-bot"}
        ).json()
        assert body["verdict"] == "review"

    def test_unknown_app_gets_strict_default(self, client):
        client.app.state.pii_detector = FakePIIDetector([])
        body = _analyze(
            client, "hello there", context={"application_id": "no-such-app"}
        ).json()
        assert body["pipeline"]["policy"]["note"].startswith("no policy")

    def test_no_context_skips_policy(self, client):
        body = _analyze(client, "hello there").json()
        assert "policy" not in body["pipeline"]


class TestSafetyVerdictParsing:
    """Unit tests for LlamaGuardClassifier._parse (no model needed)."""

    def _classifier(self):
        c = LlamaGuardClassifier.__new__(LlamaGuardClassifier)
        c.model_name = "fake"
        return c

    def test_safe(self):
        result = self._classifier()._parse("safe")
        assert result.ran and result.safe is True

    def test_unsafe_with_categories(self):
        result = self._classifier()._parse("unsafe\nS1,S9")
        assert result.safe is False
        assert result.categories == ["violent_crimes", "indiscriminate_weapons"]

    def test_unsafe_unknown_code_passes_through(self):
        result = self._classifier()._parse("unsafe\nS99")
        assert result.categories == ["S99"]

    def test_unparseable_fails_safe(self):
        result = self._classifier()._parse("I think this is fine")
        assert result.ran is False
        assert result.safe is None

    def test_all_flagged_categories_excluded_is_safe(self):
        result = self._classifier()._parse("unsafe\nS7", excluded={"S7"})
        assert result.safe is True
        assert result.note == "ignored_categories: privacy"

    def test_partially_excluded_still_blocks_on_the_rest(self):
        result = self._classifier()._parse("unsafe\nS1,S7", excluded={"S7"})
        assert result.safe is False
        assert result.categories == ["violent_crimes"]

    def test_excluded_categories_parsed_from_config(self, monkeypatch):
        from app.config import settings

        monkeypatch.setattr(settings, "safety_excluded_categories", "s6, S13,")
        assert LlamaGuardClassifier().excluded_categories == ["S6", "S13"]

    def test_no_excluded_categories_by_default(self):
        assert LlamaGuardClassifier().excluded_categories == []
