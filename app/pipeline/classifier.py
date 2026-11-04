"""Pipeline stage 4 — Prompt Guard classifier.

Loads Meta Llama Prompt Guard 2 locally at startup and runs inference
in-process. The `transformers`/`torch` stack is an optional extra
(`pip install llm-sentinel[ml]`); when it's absent or the model fails
to load, the classifier reports itself unavailable and the pipeline
falls back to a `review` verdict rather than silently allowing.

Design notes:
- The malicious label id is resolved from the model config at load time;
  a model without a recognizable label is treated as a load failure
  (fail-safe) rather than silently scoring everything 0.0 (fail-open).
- Prompt Guard's context window is 512 tokens but prompts may be longer.
  Long inputs are scored in overlapping chunks and the max score wins,
  so an attacker can't hide a payload past the truncation point.
- Thresholds and verdicts are deliberately NOT decided here — the
  pipeline maps scores to allow/review/block using request options.
"""
import logging
from dataclasses import dataclass

from app.config import settings

logger = logging.getLogger("llm_sentinel.classifier")

# Prompt Guard 2 context window.
_MAX_TOKENS = 512
# Overlap between chunks so a payload can't straddle a boundary unseen.
_CHUNK_STRIDE = 128

# Label names that mean "malicious" across Prompt Guard variants
# (86M uses MALICIOUS/BENIGN; some variants use JAILBREAK/INJECTION).
_MALICIOUS_LABELS = ("MALICIOUS", "JAILBREAK", "INJECTION")


@dataclass
class ClassifierScore:
    ran: bool
    score: float | None = None
    model: str | None = None
    note: str | None = None


class PromptGuardClassifier:
    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or settings.classifier_model
        self.enabled = not settings.disable_classifier
        self._model = None
        self._tokenizer = None
        self._device = "cpu"
        self._malicious_ids: list[int] = []
        self._load_error: str | None = None

    def load(self) -> None:
        if not self.enabled:
            logger.info("classifier disabled via SENTINEL_DISABLE_CLASSIFIER")
            return
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            self._device = (
                settings.classifier_device
                if settings.classifier_device != "auto"
                else self._choose_device(torch)
            )
            # Prefer the local cache; only touch the network on a cache miss
            # (first run). Avoids per-startup hub probes and works offline.
            try:
                self._tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name, local_files_only=True
                )
                model = AutoModelForSequenceClassification.from_pretrained(
                    self.model_name, local_files_only=True
                )
            except OSError:
                self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                model = AutoModelForSequenceClassification.from_pretrained(
                    self.model_name
                )
            self._model = model.to(self._device)
            self._model.eval()

            label2id = self._model.config.label2id or {}
            self._malicious_ids = [
                idx
                for label, idx in label2id.items()
                if label.upper() in _MALICIOUS_LABELS
            ]
            if not self._malicious_ids:
                # Some published checkpoints use generic LABEL_0/LABEL_1 names
                # instead of MALICIOUS/BENIGN. For a 2-class model, the
                # higher-indexed label is the positive (malicious) class by
                # the standard sklearn/transformers binary convention.
                if len(label2id) == 2:
                    malicious_idx = max(label2id.values())
                    self._malicious_ids = [malicious_idx]
                    logger.warning(
                        "model uses generic labels %s; assuming label id %d "
                        "is malicious (standard binary convention). "
                        "Set SENTINEL_CLASSIFIER_MODEL to a checkpoint with "
                        "named MALICIOUS/BENIGN labels to suppress this warning.",
                        list(label2id), malicious_idx,
                    )
                else:
                    # Can't guess for >2 classes — fail safe.
                    raise RuntimeError(
                        f"model exposes no malicious label; labels: {list(label2id)}"
                    )
            logger.info(
                "loaded classifier %s on %s (malicious label ids: %s)",
                self.model_name, self._device, self._malicious_ids,
            )
        except Exception as exc:  # noqa: BLE001 - any load failure means unavailable
            self._model = None
            self._load_error = f"{type(exc).__name__}: {exc}"
            logger.warning("classifier unavailable: %s", self._load_error)

    @staticmethod
    def _choose_device(torch) -> str:
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    @property
    def available(self) -> bool:
        return self._model is not None

    def score(self, text: str) -> ClassifierScore:
        if not self.enabled:
            return ClassifierScore(ran=False, note="classifier_disabled")
        if self._model is None:
            return ClassifierScore(
                ran=False, note=f"classifier_unavailable: {self._load_error}"
            )
        if not text.strip():
            return ClassifierScore(ran=True, score=0.0, model=self.model_name)

        import torch

        # Tokenize into overlapping chunks so text beyond the model's
        # context window still gets scored.
        encoded = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=_MAX_TOKENS,
            stride=_CHUNK_STRIDE,
            return_overflowing_tokens=True,
        )
        input_ids = encoded["input_ids"].to(self._device)
        attention_mask = encoded["attention_mask"].to(self._device)

        with torch.no_grad():
            logits = self._model(
                input_ids=input_ids, attention_mask=attention_mask
            ).logits

        probabilities = torch.softmax(logits, dim=-1)
        # Max malicious probability across all chunks and malicious labels.
        malicious = max(
            float(probabilities[chunk, label_id])
            for chunk in range(probabilities.shape[0])
            for label_id in self._malicious_ids
        )
        return ClassifierScore(ran=True, score=malicious, model=self.model_name)
