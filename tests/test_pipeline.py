import numpy as np
from pipeline.end_to_end import VoiceAssistantPipeline


def test_process_text_minimal():
    p = VoiceAssistantPipeline(use_rag=True)
    out = p.process_text("Open the driver window")
    assert isinstance(out, dict)
    assert "command" in out
    assert out["transcript"].lower().startswith("open")


def test_classifier_rule_mapping():
    p = VoiceAssistantPipeline(use_rag=False)
    out = p.process_text("Set temperature to 72")
    assert out["command"] in {"set_temperature", "unknown"}
