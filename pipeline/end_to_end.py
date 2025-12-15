"""End-to-end inference pipeline.

Wires STT, NLU, command classification, and optional RAG context.
"""

import logging
from typing import Dict, Any, Optional

from models import VoskSTTEngine, QuantizedNLUPipeline, CommandClassifier, VehicleRAG

logger = logging.getLogger(__name__)


class VoiceAssistantPipeline:
    """Complete in-car voice assistant pipeline."""

    def __init__(
        self,
        use_rag: bool = True,
        nlu_model_name: str = "distilbert-base-uncased-finetuned-sst-2-english",
        classifier_type: str = "rule",  # 'rule' | 'svm'
        rag_type: str = "kb",  # 'kb' | 'faiss'
    ) -> None:
        """Initialize and wire all components.

        Args:
            use_rag: Whether to enable RAG context retrieval
            nlu_model_name: Hugging Face model id for NLU
        """
        self.stt = VoskSTTEngine()
        self.nlu = QuantizedNLUPipeline(model_name=nlu_model_name)

        # Classifier selection with safe fallback
        self.classifier = CommandClassifier()
        if classifier_type == "svm":
            try:
                from models.command_classifier_svm import SVMCommandClassifier
                svm = SVMCommandClassifier()
                # Use only if it provides the same interface; otherwise fallback
                if hasattr(svm, "classify"):
                    self.classifier = svm
                    logger.info("Using SVMCommandClassifier")
                else:
                    logger.warning("SVMCommandClassifier has no classify(), falling back to rule-based")
            except Exception as e:
                logger.warning("Falling back to rule-based CommandClassifier: %s", e)

        # RAG selection with safe fallback
        self.rag = VehicleRAG() if use_rag else None
        if use_rag and rag_type == "faiss":
            try:
                from models.rag_faiss import FAISSVehicleRAG
                self.rag = FAISSVehicleRAG()
                logger.info("Using FAISSVehicleRAG")
            except Exception as e:
                logger.warning("Falling back to KB VehicleRAG: %s", e)
        logger.info("VoiceAssistantPipeline initialized")

    def process_audio(self, audio_data: bytes) -> Dict[str, Any]:
        """Process raw audio bytes through the full pipeline.

        Args:
            audio_data: Raw audio bytes

        Returns:
            Dictionary containing transcript, intent, command, confidence, and context
        """
        # STT
        stt_result = self.stt.transcribe_stream([audio_data])
        transcript = getattr(stt_result, "text", "") or ""

        # NLU
        nlu = self.nlu.process(transcript) if transcript else {"label": "unknown", "score": 0.0}

        # Command classification
        classification = self.classifier.classify(
            intent_label=nlu.get("label", "unknown"),
            confidence=float(nlu.get("score", 0.0) or 0.0),
            text=transcript,
        )

        # Optional RAG context
        context: Dict[str, Any] = {}
        if self.rag:
            context = self.rag.retrieve_context(
                query=transcript,
                command=str(classification.get("command", "unknown")),
            )

        return {
            "transcript": transcript,
            "nlu": nlu,
            "command": classification.get("command", "unknown"),
            "parameters": classification.get("parameters", {}),
            "confidence": float(classification.get("confidence", 0.0) or 0.0),
            "context": context,
        }

    def process_text(self, text: str) -> Dict[str, Any]:
        """Bypass STT and process plain text for testing or CLI use."""
        nlu = self.nlu.process(text) if text else {"label": "unknown", "score": 0.0}
        classification = self.classifier.classify(
            intent_label=nlu.get("label", "unknown"),
            confidence=float(nlu.get("score", 0.0) or 0.0),
            text=text,
        )
        context: Dict[str, Any] = {}
        if self.rag:
            context = self.rag.retrieve_context(query=text, command=str(classification.get("command", "unknown")))
        return {
            "transcript": text,
            "nlu": nlu,
            "command": classification.get("command", "unknown"),
            "parameters": classification.get("parameters", {}),
            "confidence": float(classification.get("confidence", 0.0) or 0.0),
            "context": context,
        }
