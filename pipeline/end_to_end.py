"""End-to-end inference pipeline.

Wires STT, NLU, command classification, and optional RAG context.
"""

import logging
from typing import Dict, Any, Optional
import json

from models import VoskSTTEngine, QuantizedNLUPipeline, CommandClassifier, VehicleRAG

logger = logging.getLogger(__name__)


class VoiceAssistantPipeline:
    """Complete in-car voice assistant pipeline."""

    def __init__(
        self,
        use_rag: bool = True,
        nlu_model_name: str = "microsoft/phi-2",  # Default matches training model
        classifier_type: str = "rule",  # 'rule' | 'svm'
        rag_type: str = "kb",  # 'kb' | 'faiss'
        adapter_path: Optional[str] = None,  # Path to LoRA adapter for domain adaptation
    ) -> None:
        """Initialize and wire all components.

        Args:
            use_rag: Whether to enable RAG context retrieval
            nlu_model_name: Hugging Face model id for NLU (default: microsoft/phi-2)
            classifier_type: Classifier to use ('rule' or 'svm')
            rag_type: RAG backend ('kb' or 'faiss')
            adapter_path: Optional path to LoRA adapter checkpoint for domain-adapted inference
        """
        self.stt = VoskSTTEngine()
        self.nlu = QuantizedNLUPipeline(model_name=nlu_model_name, adapter_path=adapter_path)

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

    @staticmethod
    def _parse_generated_assistant(generated_text: str) -> Dict[str, Any]:
        """Parse Command and Parameters from generated assistant text.

        Expects patterns like:
        "Assistant: Intent=..., Command=<cmd>, Parameters={...}"
        """
        result: Dict[str, Any] = {"command": "unknown", "parameters": {}}
        if not generated_text:
            return result
        try:
            # Extract Command=
            if "Command=" in generated_text:
                cmd_part = generated_text.split("Command=")[-1]
                cmd_token = cmd_part.split(",")[0].split("\n")[0].strip()
                if cmd_token:
                    result["command"] = cmd_token
            # Extract Parameters={...}
            if "Parameters=" in generated_text:
                params_part = generated_text.split("Parameters=")[-1]
                # Up to newline
                params_str = params_part.split("\n")[0].strip()
                # Ensure we only parse the JSON object
                if params_str:
                    # Some generations may append trailing text; cut after matching braces
                    if "}" in params_str:
                        params_str = params_str[: params_str.find("}") + 1]
                    try:
                        parsed = json.loads(params_str)
                        if isinstance(parsed, dict):
                            result["parameters"] = parsed
                    except Exception:
                        pass
        except Exception:
            pass
        return result

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
        nlu_cmd = "unknown"
        nlu_params: Dict[str, Any] = {}
        if isinstance(nlu, dict) and nlu.get("generated"):
            parsed = self._parse_generated_assistant(str(nlu.get("generated")))
            nlu_cmd = parsed.get("command", "unknown")
            nlu_params = parsed.get("parameters", {})

        # Command classification
        classification = self.classifier.classify(
            intent_label=nlu.get("label", "unknown"),
            confidence=float(nlu.get("score", 0.0) or 0.0),
            text=transcript,
        )

        # Prefer NLU-derived command if available; merge parameters
        command = nlu_cmd if nlu_cmd and nlu_cmd != "unknown" else classification.get("command", "unknown")
        parameters = classification.get("parameters", {}) or {}
        if nlu_params:
            # Merge, NLU takes precedence for keys
            parameters = {**parameters, **nlu_params}

        # Optional RAG context
        context: Dict[str, Any] = {}
        if self.rag:
            context = self.rag.retrieve_context(
                query=transcript,
                command=str(command),
            )

        return {
            "transcript": transcript,
            "nlu": nlu,
            "command": command,
            "parameters": parameters,
            "confidence": float(classification.get("confidence", 0.0) or 0.0),
            "context": context,
        }

    def process_text(self, text: str) -> Dict[str, Any]:
        """Bypass STT and process plain text for testing or CLI use."""
        nlu = self.nlu.process(text) if text else {"label": "unknown", "score": 0.0}
        nlu_cmd = "unknown"
        nlu_params: Dict[str, Any] = {}
        if isinstance(nlu, dict) and nlu.get("generated"):
            parsed = self._parse_generated_assistant(str(nlu.get("generated")))
            nlu_cmd = parsed.get("command", "unknown")
            nlu_params = parsed.get("parameters", {})
        classification = self.classifier.classify(
            intent_label=nlu.get("label", "unknown"),
            confidence=float(nlu.get("score", 0.0) or 0.0),
            text=text,
        )
        command = nlu_cmd if nlu_cmd and nlu_cmd != "unknown" else classification.get("command", "unknown")
        parameters = classification.get("parameters", {}) or {}
        if nlu_params:
            parameters = {**parameters, **nlu_params}
        context: Dict[str, Any] = {}
        if self.rag:
            context = self.rag.retrieve_context(query=text, command=str(command))
        return {
            "transcript": text,
            "nlu": nlu,
            "command": command,
            "parameters": parameters,
            "confidence": float(classification.get("confidence", 0.0) or 0.0),
            "context": context,
        }


class InCarVoiceAssistant(VoiceAssistantPipeline):
    """Backward-compatible alias for CI/imports."""

    pass
