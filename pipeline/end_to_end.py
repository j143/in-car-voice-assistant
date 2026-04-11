"""End-to-end inference pipeline.

Wires STT, NLU, command classification, and optional RAG context.

NLU backend selection (``nlu_type`` parameter):
  ``"auto"`` (default) — use CPU TF-IDF NLU when torch is unavailable,
  otherwise fall back to ``QuantizedNLUPipeline``.
  ``"cpu"`` — always use the lightweight TF-IDF NLU (no torch required).
  ``"transformer"`` — always use ``QuantizedNLUPipeline`` (torch required).
"""

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def _torch_available() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


class VoiceAssistantPipeline:
    """Complete in-car voice assistant pipeline."""

    def __init__(
        self,
        use_rag: bool = True,
        nlu_model_name: str = "distilbert-base-uncased-finetuned-sst-2-english",
        nlu_type: str = "auto",          # "auto" | "cpu" | "transformer"
        classifier_type: str = "rule",   # "rule" | "svm"
        rag_type: str = "kb",            # "kb"   | "faiss"
        adapter_path: Optional[str] = None,
    ) -> None:
        """Initialise and wire all pipeline components.

        Args:
            use_rag: Enable RAG context retrieval.
            nlu_model_name: HuggingFace model id used when nlu_type is
                ``"transformer"`` or ``"auto"`` with torch available.
            nlu_type: NLU backend — see module docstring.
            classifier_type: ``"rule"`` (default) or ``"svm"``.
            rag_type: ``"kb"`` (default) or ``"faiss"``.
            adapter_path: Optional LoRA adapter path for domain-adapted inference.
        """
        from models.stt_engine import VoskSTTEngine
        from models.command_classifier import CommandClassifier
        from models.rag_component import VehicleRAG

        self.stt = VoskSTTEngine()

        # ── NLU backend ────────────────────────────────────────────────
        use_cpu_nlu = (
            nlu_type == "cpu"
            or (nlu_type == "auto" and not _torch_available())
        )
        if use_cpu_nlu:
            from models.nlu_cpu import CpuNLU
            self.nlu = CpuNLU()
            logger.info("NLU backend: CpuNLU (TF-IDF, no torch required)")
        else:
            from models.nlu_pipeline import QuantizedNLUPipeline
            self.nlu = QuantizedNLUPipeline(
                model_name=nlu_model_name,
                adapter_path=adapter_path,
            )
            logger.info("NLU backend: QuantizedNLUPipeline (%s)", nlu_model_name)

        # ── Classifier ─────────────────────────────────────────────────
        self.classifier = CommandClassifier()
        if classifier_type == "svm":
            try:
                from models.command_classifier_svm import SVMCommandClassifier
                svm = SVMCommandClassifier()
                if hasattr(svm, "classify"):
                    self.classifier = svm
                    logger.info("Using SVMCommandClassifier")
                else:
                    logger.warning("SVMCommandClassifier has no classify(); falling back to rule-based")
            except Exception as e:
                logger.warning("Falling back to rule-based CommandClassifier: %s", e)

        # ── RAG ────────────────────────────────────────────────────────
        self.rag = VehicleRAG() if use_rag else None
        if use_rag and rag_type == "faiss":
            try:
                from models.rag_faiss import FAISSVehicleRAG
                self.rag = FAISSVehicleRAG()
                logger.info("Using FAISSVehicleRAG")
            except Exception as e:
                logger.warning("Falling back to KB VehicleRAG: %s", e)

        logger.info("VoiceAssistantPipeline ready (nlu=%s, classifier=%s, rag=%s)",
                    "cpu" if use_cpu_nlu else "transformer", classifier_type, rag_type)

    # ── Public API ─────────────────────────────────────────────────────

    def process_audio(self, audio_data: bytes) -> Dict[str, Any]:
        """Process raw PCM-16 audio bytes through the full pipeline."""
        from models.stt_engine import TranscriptionResult
        stt_result = self.stt.transcribe_stream([audio_data])
        transcript = getattr(stt_result, "text", "") or ""
        return self._run(transcript)

    def process_text(self, text: str) -> Dict[str, Any]:
        """Bypass STT and process plain text (for CLI / testing)."""
        return self._run(text)

    # ── Internal ───────────────────────────────────────────────────────

    def _run(self, text: str) -> Dict[str, Any]:
        if not text:
            return {
                "transcript": text,
                "nlu": {"label": "unknown", "score": 0.0},
                "command": "unknown",
                "parameters": {},
                "confidence": 0.0,
                "context": {},
            }

        nlu = self.nlu.process(text)
        classification = self.classifier.classify(
            intent_label=nlu.get("label", "unknown"),
            confidence=float(nlu.get("score", 0.0) or 0.0),
            text=text,
        )
        context: Dict[str, Any] = {}
        if self.rag:
            context = self.rag.retrieve_context(
                query=text,
                command=str(classification.get("command", "unknown")),
            )
        return {
            "transcript": text,
            "nlu": nlu,
            "command": classification.get("command", "unknown"),
            "parameters": classification.get("parameters", {}),
            "confidence": float(classification.get("confidence", 0.0) or 0.0),
            "context": context,
        }


class InCarVoiceAssistant(VoiceAssistantPipeline):
    """Backward-compatible alias."""
    pass
