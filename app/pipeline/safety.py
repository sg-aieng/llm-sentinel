"""Output pipeline — Llama Guard 3 content-safety classifier.

Unlike Prompt Guard (an encoder that emits a probability), Llama Guard 3
is a causal LM: it reads the conversation through its chat template and
*generates* a verdict — "safe", or "unsafe" followed by category codes
(S1..S14). There is no score, so no thresholds: safe → allow,
unsafe → block, anything else → fail-safe review.

Same lifecycle doctrine as the other models: loaded once at startup as
an optional `[ml]` dependency, `local_files_only` cache-first loading,
and graceful degradation — when the model is unavailable the stage
reports itself not-ran and the pipeline downgrades to `review`.

Note: the default model (meta-llama/Llama-Guard-3-1B) is gated on
Hugging Face; request access or point SENTINEL_SAFETY_MODEL at an
ungated mirror.
"""
import logging
from dataclasses import dataclass, field

from app.config import settings

logger = logging.getLogger("llm_sentinel.safety")

# Llama Guard 3 hazard taxonomy (MLCommons). Codes the model may emit
# after an "unsafe" verdict, mapped to readable category names.
CATEGORY_NAMES = {
    "S1": "violent_crimes",
    "S2": "non_violent_crimes",
    "S3": "sex_related_crimes",
    "S4": "child_sexual_exploitation",
    "S5": "defamation",
    "S6": "specialized_advice",
    "S7": "privacy",
    "S8": "intellectual_property",
    "S9": "indiscriminate_weapons",
    "S10": "hate",
    "S11": "suicide_and_self_harm",
    "S12": "sexual_content",
    "S13": "elections",
    "S14": "code_interpreter_abuse",
}


@dataclass
class SafetyResult:
    ran: bool
    safe: bool | None = None
    categories: list[str] = field(default_factory=list)
    model: str | None = None
    note: str | None = None


class LlamaGuardClassifier:
    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or settings.safety_model
        self.enabled = not settings.disable_safety
        self.max_new_tokens = settings.safety_max_new_tokens
        # Hazard categories to exclude from enforcement (comma-separated
        # codes in config, e.g. "S6,S13").
        self.excluded_categories = [
            c.strip().upper()
            for c in settings.safety_excluded_categories.split(",")
            if c.strip()
        ]
        self._model = None
        self._tokenizer = None
        self._device = "cpu"
        self._load_error: str | None = None

    def load(self) -> None:
        if not self.enabled:
            logger.info("safety classifier disabled via SENTINEL_DISABLE_SAFETY")
            return
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self._device = (
                settings.safety_device
                if settings.safety_device != "auto"
                else self._choose_device(torch)
            )
            # Prefer the local cache; only touch the network on a cache miss
            # (first run). Avoids per-startup hub probes and works offline.
            try:
                self._tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name, local_files_only=True
                )
                model = AutoModelForCausalLM.from_pretrained(
                    self.model_name, local_files_only=True, torch_dtype="auto"
                )
            except OSError:
                self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                model = AutoModelForCausalLM.from_pretrained(
                    self.model_name, torch_dtype="auto"
                )
            if self._tokenizer.chat_template is None:
                # Fail-safe: without the Llama Guard template we can't build
                # the classification prompt, so verdicts would be garbage.
                raise RuntimeError("tokenizer has no chat template")
            self._model = model.to(self._device)
            self._model.eval()
            logger.info(
                "loaded safety classifier %s on %s", self.model_name, self._device
            )
        except Exception as exc:  # noqa: BLE001 - any load failure means unavailable
            self._model = None
            self._load_error = f"{type(exc).__name__}: {exc}"
            logger.warning("safety classifier unavailable: %s", self._load_error)

    @staticmethod
    def _choose_device(torch) -> str:
        """Auto-detect best available device: cuda → mps → cpu."""
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    @property
    def available(self) -> bool:
        return self._model is not None

    @property
    def load_error(self) -> str | None:
        return self._load_error

    def classify(
        self,
        output: str,
        prompt: str | None = None,
        extra_excluded_categories: list[str] | None = None,
    ) -> SafetyResult:
        """Classify a model response, optionally with the prompt it answered.

        Llama Guard assesses the LAST turn of the conversation, so the
        response is placed in the assistant turn and the original prompt
        (when the caller supplies it) in the user turn for context.

        `extra_excluded_categories` adds per-request exclusions on top of
        the configured ones — e.g. the pipeline drops S7 (privacy) when
        the application's policy already handled PII.
        """
        if not self.enabled:
            return SafetyResult(ran=False, note="safety_disabled")
        if self._model is None:
            return SafetyResult(
                ran=False, note=f"safety_unavailable: {self._load_error}"
            )

        try:
            return self._classify(output, prompt, extra_excluded_categories or [])
        except Exception as exc:  # noqa: BLE001 - fail safe, never 500 the request
            logger.warning("safety classification failed: %s", exc)
            return SafetyResult(
                ran=False,
                model=self.model_name,
                note=f"safety_error: {type(exc).__name__}: {exc}",
            )

    def _classify(
        self, output: str, prompt: str | None, extra_excluded: list[str]
    ) -> SafetyResult:
        import torch

        # Structured content parts, per Meta's reference implementation:
        # the Llama Guard 3 chat template iterates content as a list of
        # typed parts, not a plain string.
        conversation = [
            {"role": "user", "content": [{"type": "text", "text": prompt or ""}]},
            {"role": "assistant", "content": [{"type": "text", "text": output}]},
        ]
        # Excluded categories are filtered from the VERDICT, not removed
        # from the template: dropping categories from the classification
        # prompt makes the (small) model re-attribute the finding to an
        # unrelated category instead of passing the text.
        excluded = {*self.excluded_categories, *extra_excluded}
        # return_dict=True gives a stable {input_ids, attention_mask} shape
        # across transformers v4 (bare tensor otherwise) and v5 (BatchEncoding).
        encoded = self._tokenizer.apply_chat_template(
            conversation, return_tensors="pt", return_dict=True
        )
        input_ids = encoded["input_ids"].to(self._device)
        attention_mask = encoded.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self._device)

        with torch.no_grad():
            generated = self._model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        verdict_text = self._tokenizer.decode(
            generated[0][input_ids.shape[-1]:], skip_special_tokens=True
        ).strip()

        return self._parse(verdict_text, excluded)

    def _parse(
        self, verdict_text: str, excluded: set[str] | None = None
    ) -> SafetyResult:
        excluded = excluded or set()
        lines = [ln.strip() for ln in verdict_text.splitlines() if ln.strip()]
        head = lines[0].lower() if lines else ""
        if head == "safe":
            return SafetyResult(ran=True, safe=True, model=self.model_name)
        if head == "unsafe":
            codes = []
            if len(lines) > 1:
                codes = [
                    c.upper()
                    for c in lines[1].replace(" ", "").split(",")
                    if c
                ]
            enforced = [c for c in codes if c not in excluded]
            if codes and not enforced:
                # Everything the model flagged is excluded from enforcement.
                return SafetyResult(
                    ran=True,
                    safe=True,
                    model=self.model_name,
                    note="ignored_categories: "
                    + ",".join(CATEGORY_NAMES.get(c, c) for c in codes),
                )
            return SafetyResult(
                ran=True,
                safe=False,
                categories=[CATEGORY_NAMES.get(c, c) for c in enforced],
                model=self.model_name,
            )
        # Fail-safe: an unparseable verdict must not be treated as safe.
        return SafetyResult(
            ran=False,
            model=self.model_name,
            note=f"unparseable_verdict: {verdict_text[:80]!r}",
        )
