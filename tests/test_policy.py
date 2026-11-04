"""Unit tests for the policy engine, PII redaction, and identity provider."""
import pytest
from pathlib import Path

from app.identity import Identity, StaticIdentityProvider
from app.pipeline.pii import PIIEntity, redact
from app.pipeline.policy import PolicyEngine

POLICIES_DIR = Path(__file__).resolve().parents[1] / "policies"


class FakePIIDetector:
    """Returns canned entities; None mimics an unavailable model."""

    def __init__(self, entities: list[PIIEntity] | None):
        self._entities = entities
        self.load_error = "test" if entities is None else None

    @property
    def available(self) -> bool:
        return self._entities is not None

    def detect(self, text: str) -> list[PIIEntity] | None:
        return self._entities


def make_engine() -> PolicyEngine:
    return PolicyEngine(policies_dir=POLICIES_DIR)


NO_PII = FakePIIDetector([])


class TestCapabilities:
    def test_allowed_capability_passes(self):
        d = make_engine().evaluate(
            "support-bot", "hello", ["web_search"], ["default"], NO_PII
        )
        assert d.verdict is None
        assert d.capability_violations == []

    def test_blocked_capability_blocks(self):
        d = make_engine().evaluate(
            "support-bot", "hello", ["code_execution"], ["default"], NO_PII
        )
        assert d.verdict == "block"
        assert d.blocked_by == "capabilities"
        assert d.capability_violations == ["code_execution"]

    def test_unlisted_capability_blocks(self):
        d = make_engine().evaluate(
            "support-bot", "hello", ["send_email"], ["default"], NO_PII
        )
        assert d.verdict == "block"

    def test_admin_role_unlocks_code_execution(self):
        d = make_engine().evaluate(
            "support-bot", "hello", ["code_execution"], ["admin"], NO_PII
        )
        assert d.verdict is None

    def test_admin_still_denied_file_write(self):
        d = make_engine().evaluate(
            "support-bot", "hello", ["file_write"], ["admin"], NO_PII
        )
        assert d.verdict == "block"

    def test_no_capabilities_requested_passes(self):
        d = make_engine().evaluate("support-bot", "hello", [], ["default"], NO_PII)
        assert d.verdict is None


class TestTopics:
    def test_blocked_topic_keyword(self):
        d = make_engine().evaluate(
            "support-bot", "What dosage of ibuprofen is safe?", [], ["default"], NO_PII
        )
        assert d.verdict == "block"
        assert d.blocked_by == "topics"
        assert "medical_advice" in d.topic_matches

    def test_keyword_match_is_case_insensitive(self):
        d = make_engine().evaluate(
            "support-bot", "Tell me about ACME Corp pricing", [], ["default"], NO_PII
        )
        assert d.verdict == "block"
        assert "competitors" in d.topic_matches

    def test_clean_topic_passes(self):
        d = make_engine().evaluate(
            "support-bot", "How do I reset my password?", [], ["default"], NO_PII
        )
        assert d.topic_matches == []


class TestPII:
    def test_redact_action(self):
        text = "Contact john@example.com or call 555-1234"
        detector = FakePIIDetector([
            PIIEntity("EMAIL", "john@example.com", 8, 24),
            PIIEntity("PHONE", "555-1234", 33, 41),
        ])
        d = make_engine().evaluate("support-bot", text, [], ["default"], detector)
        assert d.verdict is None
        assert d.pii_found == ["EMAIL", "PHONE"]
        assert "[EMAIL_1]" in d.redacted_prompt
        assert "[PHONE_1]" in d.redacted_prompt
        assert "john@example.com" not in d.redacted_prompt

    def test_no_pii_found(self):
        d = make_engine().evaluate("support-bot", "hello world", [], ["default"], NO_PII)
        assert d.pii_found == []
        assert d.redacted_prompt is None

    def test_detector_unavailable_forces_review(self):
        d = make_engine().evaluate(
            "support-bot", "hello", [], ["default"], FakePIIDetector(None)
        )
        assert d.verdict == "review"
        assert "unavailable" in d.note

    def test_entity_filter_respected(self):
        # USERNAME is not in support-bot's entities list — must be ignored.
        detector = FakePIIDetector([PIIEntity("USERNAME", "jdoe", 0, 4)])
        d = make_engine().evaluate("support-bot", "jdoe here", [], ["default"], detector)
        assert d.pii_found == []
        assert d.redacted_prompt is None


class TestStrictDefault:
    def test_unknown_app_gets_strict_default(self):
        d = make_engine().evaluate(
            "no-such-app", "hello", ["web_search"], ["default"], NO_PII
        )
        assert d.verdict == "block"
        assert "strict default" in d.note

    def test_unknown_app_no_capabilities_passes(self):
        d = make_engine().evaluate("no-such-app", "hello", [], ["default"], NO_PII)
        assert d.verdict is None


class TestRedactFunction:
    def test_same_value_same_placeholder(self):
        text = "a@b.com wrote to a@b.com"
        entities = [
            PIIEntity("EMAIL", "a@b.com", 0, 7),
            PIIEntity("EMAIL", "a@b.com", 17, 24),
        ]
        redacted, found = redact(text, entities)
        assert redacted == "[EMAIL_1] wrote to [EMAIL_1]"
        assert found == ["EMAIL"]

    def test_different_values_numbered(self):
        text = "a@b.com and c@d.com"
        entities = [
            PIIEntity("EMAIL", "a@b.com", 0, 7),
            PIIEntity("EMAIL", "c@d.com", 12, 19),
        ]
        redacted, _ = redact(text, entities)
        assert redacted == "[EMAIL_1] and [EMAIL_2]"


class TestIdentity:
    def test_unknown_user_gets_default_role(self):
        provider = StaticIdentityProvider(identities_path=Path("/nonexistent"))
        identity = provider.resolve("anyone")
        assert identity.roles == ["default"]

    def test_known_user_from_file(self, tmp_path):
        f = tmp_path / "identities.yaml"
        f.write_text("users:\n  u1:\n    roles: [admin]\n")
        provider = StaticIdentityProvider(identities_path=f)
        assert provider.resolve("u1").roles == ["admin"]
        assert provider.resolve("u2").roles == ["default"]


def _engine_from_doc(tmp_path, doc: dict) -> PolicyEngine:
    """Write a single-doc policy to a temp dir and load it."""
    import yaml
    p = tmp_path / "app.yaml"
    p.write_text(yaml.dump(doc))
    return PolicyEngine(policies_dir=tmp_path)


class TestPolicyValidation:
    """_validate_policy raises ValueError for every malformed structure."""

    # -- id ----------------------------------------------------------------

    def test_id_must_be_string(self, tmp_path):
        with pytest.raises(ValueError, match="'id' must be a string"):
            _engine_from_doc(tmp_path, {"id": 123})

    # -- capabilities ------------------------------------------------------

    def test_capabilities_must_be_mapping(self, tmp_path):
        with pytest.raises(ValueError, match="'capabilities' must be a mapping"):
            _engine_from_doc(tmp_path, {"capabilities": ["web_search"]})

    def test_capabilities_allowed_must_be_list(self, tmp_path):
        with pytest.raises(ValueError, match="capabilities.allowed must be a list"):
            _engine_from_doc(tmp_path, {"capabilities": {"allowed": "web_search"}})

    def test_capabilities_blocked_must_be_list(self, tmp_path):
        with pytest.raises(ValueError, match="capabilities.blocked must be a list"):
            _engine_from_doc(tmp_path, {"capabilities": {"blocked": "code_execution"}})

    def test_capabilities_allowed_elements_must_be_strings(self, tmp_path):
        with pytest.raises(ValueError, match=r"capabilities\.allowed\[0\] must be a string"):
            _engine_from_doc(tmp_path, {"capabilities": {"allowed": [42]}})

    def test_capabilities_blocked_elements_must_be_strings(self, tmp_path):
        with pytest.raises(ValueError, match=r"capabilities\.blocked\[0\] must be a string"):
            _engine_from_doc(tmp_path, {"capabilities": {"blocked": [{"name": "bad"}]}})

    def test_capabilities_unknown_key(self, tmp_path):
        with pytest.raises(ValueError, match="unknown keys"):
            _engine_from_doc(tmp_path, {"capabilities": {"allowed": [], "extra": []}})

    # -- roles -------------------------------------------------------------

    def test_roles_must_be_mapping(self, tmp_path):
        with pytest.raises(ValueError, match="'roles' must be a mapping"):
            _engine_from_doc(tmp_path, {"roles": ["admin"]})

    def test_role_def_must_be_mapping(self, tmp_path):
        with pytest.raises(ValueError, match=r"roles\['admin'\] must be a mapping"):
            _engine_from_doc(tmp_path, {"roles": {"admin": "superuser"}})

    def test_role_unknown_key(self, tmp_path):
        with pytest.raises(ValueError, match="unknown keys"):
            _engine_from_doc(
                tmp_path,
                {"roles": {"admin": {"capabilities": {}, "extra": True}}},
            )

    def test_role_capabilities_must_be_mapping(self, tmp_path):
        with pytest.raises(ValueError, match=r"must be a mapping"):
            _engine_from_doc(
                tmp_path,
                {"roles": {"admin": {"capabilities": ["web_search"]}}},
            )

    def test_role_capabilities_allowed_elements_must_be_strings(self, tmp_path):
        with pytest.raises(ValueError, match="must be a string"):
            _engine_from_doc(
                tmp_path,
                {"roles": {"admin": {"capabilities": {"allowed": [99]}}}},
            )

    # -- pii ---------------------------------------------------------------

    def test_pii_entities_must_be_list(self, tmp_path):
        with pytest.raises(ValueError, match="pii.entities must be a list"):
            _engine_from_doc(tmp_path, {"pii": {"action": "redact", "entities": "EMAIL"}})

    def test_pii_entities_elements_must_be_strings(self, tmp_path):
        with pytest.raises(ValueError, match=r"pii\.entities\[0\] must be a string"):
            _engine_from_doc(tmp_path, {"pii": {"action": "redact", "entities": [1, 2]}})

    # -- topics ------------------------------------------------------------

    def test_topic_id_must_be_string(self, tmp_path):
        with pytest.raises(ValueError, match=r"topics\.blocked\[0\]\.id must be a string"):
            _engine_from_doc(
                tmp_path,
                {"topics": {"blocked": [{"id": 99, "keywords": ["test"]}]}},
            )

    def test_topic_keyword_elements_must_be_strings(self, tmp_path):
        with pytest.raises(ValueError, match=r"topics\.blocked\[0\]\.keywords\[1\] must be a string"):
            _engine_from_doc(
                tmp_path,
                {"topics": {"blocked": [{"id": "bad-topic", "keywords": ["ok", 42]}]}},
            )

    # -- valid docs pass ---------------------------------------------------

    def test_valid_minimal_doc_loads(self, tmp_path):
        engine = _engine_from_doc(tmp_path, {})
        assert isinstance(engine, PolicyEngine)

    def test_valid_full_doc_loads(self, tmp_path):
        doc = {
            "id": "test-app",
            "capabilities": {"allowed": ["search"], "blocked": ["exec"]},
            "roles": {
                "admin": {"capabilities": {"allowed": ["search", "exec"], "blocked": []}},
            },
            "topics": {"blocked": [{"id": "off-limits", "keywords": ["secret"]}]},
            "pii": {"action": "redact", "entities": ["EMAIL", "PHONE"]},
        }
        engine = _engine_from_doc(tmp_path, doc)
        assert "test-app" in engine.policies


class TestEntityType:
    """_entity_type retypes refined spans by what the text actually is."""

    def test_email_retyped_regardless_of_model_label(self):
        from app.pipeline.pii import _entity_type

        assert _entity_type("USERNAME", "john.smith@example.com") == "EMAIL"
        assert _entity_type("GIVENNAME", "a@b.co") == "EMAIL"

    def test_non_email_keeps_canonical_mapping(self):
        from app.pipeline.pii import _entity_type

        assert _entity_type("USERNAME", "john.smith") == "USERNAME"
        assert _entity_type("PER", "John Smith") == "PERSON"
        assert _entity_type("TELEPHONENUM", "555-0100") == "PHONE"


class TestEmailRegexFallback:
    """PIIDetector.detect() regex fallback catches emails the ML model misses."""

    def _detector_with_no_ml(self):
        """A PIIDetector whose _pipeline returns no entities (simulates model miss)."""
        from unittest.mock import MagicMock
        from app.pipeline.pii import PIIDetector

        d = PIIDetector.__new__(PIIDetector)
        d.enabled = True
        d._load_error = None
        # Pipeline that always returns an empty list (model saw nothing)
        d._pipeline = MagicMock(return_value=[])
        d._ner_pipeline = None
        return d

    def test_email_caught_when_model_returns_nothing(self):
        d = self._detector_with_no_ml()
        entities = d.detect("My email is alice@example.com — can you help me?")
        assert any(e.type == "EMAIL" and "alice@example.com" in e.text for e in entities)

    def test_email_not_duplicated_when_ml_already_found_it(self):
        from unittest.mock import MagicMock
        from app.pipeline.pii import PIIDetector

        d = PIIDetector.__new__(PIIDetector)
        d.enabled = True
        d._load_error = None
        text = "Contact alice@example.com for help."
        # ML pipeline returns the email span directly
        d._pipeline = MagicMock(return_value=[{
            "entity_group": "EMAIL",
            "start": 8,
            "end": 25,
            "score": 0.99,
        }])
        d._ner_pipeline = None
        entities = d.detect(text)
        email_entities = [e for e in entities if e.type == "EMAIL"]
        assert len(email_entities) == 1, "email must not be reported twice"

    def test_no_false_positive_for_non_email(self):
        d = self._detector_with_no_ml()
        entities = d.detect("Visit our website at example.com for more info.")
        assert not any(e.type == "EMAIL" for e in entities)
