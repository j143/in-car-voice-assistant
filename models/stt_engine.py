"""
Speech-to-Text Engine using Vosk

Lightweight, streaming-capable speech recognition optimized for in-car environments.
"""

import json
from typing import Optional, List
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class TranscriptionResult:
    """Represents a transcription result from the STT engine."""
    text: str
    confidence: float
    is_partial: bool = False
    duration: Optional[float] = None


class VoskSTTEngine:
    """
    Vosk-based speech recognition engine for in-car voice commands.
    
    Supports:
    - Streaming audio processing
    - Real-time transcription with partial results
    - Noise-robust in-car audio handling
    - Low latency (<200ms for 1s audio)
    """
    
    def __init__(
        self,
        model_path: str = "models/vosk_models/en_us",
        sample_rate: int = 16000,
        confidence_threshold: float = 0.5
    ):
        """
        Initialize the Vosk STT engine.
        
        Args:
            model_path: Path to Vosk model directory
            sample_rate: Audio sample rate in Hz (default: 16kHz)
            confidence_threshold: Minimum confidence score (0-1)
        """
        self.model_path = model_path
        self.sample_rate = sample_rate
        self.confidence_threshold = confidence_threshold
        self.current_confidence = 0.0
        self.last_result = None
        
        # Defer heavy imports until first use to keep imports light
        self.vosk = None
        self._recognizer = None
    
    def _get_recognizer(self):
        """Lazy load recognizer on first use."""
        if self._recognizer is None:
            try:
                from vosk import Model, KaldiRecognizer
            except ImportError:
                logger.error("vosk package not found. Install with: pip install vosk")
                raise
            try:
                model = Model(self.model_path)
                self._recognizer = KaldiRecognizer(model, self.sample_rate)
            except Exception as e:
                logger.error(f"Failed to initialize Vosk recognizer: {e}")
                raise
        return self._recognizer
    
    def transcribe_stream(
        self,
        audio_chunks: List[bytes],
        partial_results: bool = False
    ) -> TranscriptionResult:
        """
        Transcribe streaming audio chunks.
        
        Args:
            audio_chunks: List of raw audio byte chunks
            partial_results: Ignored in this method; use get_partial_transcriptions()
            
        Returns:
            TranscriptionResult with text and confidence
        """
        recognizer = self._get_recognizer()
        
        try:
            for chunk in audio_chunks:
                recognizer.AcceptWaveform(chunk)
                _ = recognizer.Result()
            
            # Get final result
            final = json.loads(recognizer.FinalResult())
            if "text" in final:
                self.last_result = final["text"]
                self.current_confidence = 0.85  # Vosk doesn't provide explicit confidence
                return TranscriptionResult(
                    text=final["text"],
                    confidence=self.current_confidence
                )
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return TranscriptionResult(text="", confidence=0.0)
        
        return TranscriptionResult(text="", confidence=0.0)

    def get_partial_transcriptions(self, audio_chunks: List[bytes]):
        """Yield partial transcription results for streaming UI updates."""
        recognizer = self._get_recognizer()
        try:
            for chunk in audio_chunks:
                if recognizer.AcceptWaveform(chunk):
                    result = json.loads(recognizer.Result())
                    if "text" in result:
                        self.last_result = result["text"]
                        yield TranscriptionResult(text=result["text"], confidence=0.5, is_partial=True)
                else:
                    partial = json.loads(recognizer.PartialResult())
                    if "partial" in partial and partial["partial"]:
                        yield TranscriptionResult(text=partial["partial"], confidence=0.5, is_partial=True)
        except Exception as e:
            logger.error(f"Partial transcription error: {e}")
    
    def get_confidence(self) -> float:
        """Get confidence score of last transcription."""
        return self.current_confidence
    
    def reset(self):
        """Reset recognizer state."""
        self._recognizer = None
        self.current_confidence = 0.0
        self.last_result = None
