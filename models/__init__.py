"""
In-Car Voice Assistant Models Package

Core modules for speech recognition, NLU, classification, and RAG.
"""

from .stt_engine import VoskSTTEngine
from .nlu_pipeline import QuantizedNLUPipeline
from .command_classifier import CommandClassifier
from .rag_component import VehicleRAG

__version__ = "0.1.0"
__all__ = [
    "VoskSTTEngine",
    "QuantizedNLUPipeline",
    "CommandClassifier",
    "VehicleRAG",
]
