"""Pipeline integration tests — CPU path only (no torch required)."""

import pytest
from pipeline.end_to_end import VoiceAssistantPipeline


@pytest.fixture(scope="module")
def pipeline():
    return VoiceAssistantPipeline(use_rag=True, nlu_type="cpu")


@pytest.fixture(scope="module")
def pipeline_no_rag():
    return VoiceAssistantPipeline(use_rag=False, nlu_type="cpu")


# ── Basic structure ──────────────────────────────────────────────────────────

def test_process_text_returns_dict(pipeline):
    out = pipeline.process_text("Open the driver window")
    assert isinstance(out, dict)
    assert {"transcript", "nlu", "command", "parameters", "confidence", "context"} <= out.keys()


def test_transcript_preserved(pipeline):
    text = "Navigate to the airport"
    out = pipeline.process_text(text)
    assert out["transcript"] == text


# ── Climate control ──────────────────────────────────────────────────────────

def test_temperature_command(pipeline_no_rag):
    out = pipeline_no_rag.process_text("Set temperature to 72")
    assert out["command"] == "set_temperature"


def test_temperature_parameter_extracted(pipeline_no_rag):
    out = pipeline_no_rag.process_text("Set temperature to 72")
    assert out["parameters"].get("temperature") == 72


def test_temperature_with_unit(pipeline_no_rag):
    out = pipeline_no_rag.process_text("Set cabin temperature to 22 degrees celsius")
    assert out["command"] == "set_temperature"
    assert out["parameters"].get("temperature") == 22


def test_fan_speed(pipeline_no_rag):
    out = pipeline_no_rag.process_text("Increase fan speed to level 4")
    assert out["command"] == "set_fan_speed"
    assert out["parameters"].get("fan_speed") == 4


# ── Music ────────────────────────────────────────────────────────────────────

def test_play_music(pipeline_no_rag):
    out = pipeline_no_rag.process_text("Play music via bluetooth")
    assert out["command"] == "play_music"


def test_next_track(pipeline_no_rag):
    out = pipeline_no_rag.process_text("Next track")
    assert out["command"] == "next_track"


def test_set_volume(pipeline_no_rag):
    out = pipeline_no_rag.process_text("Set volume to 60 percent")
    assert out["command"] == "set_volume"
    assert out["parameters"].get("level") == 60


# ── Navigation ───────────────────────────────────────────────────────────────

def test_navigate(pipeline_no_rag):
    out = pipeline_no_rag.process_text("Navigate to nearest charging station")
    assert out["command"] == "navigate"


def test_navigate_destination(pipeline_no_rag):
    out = pipeline_no_rag.process_text("Navigate to the airport")
    assert out["parameters"].get("destination") is not None


# ── Error / diagnostics ──────────────────────────────────────────────────────

def test_dtc_error_code(pipeline_no_rag):
    out = pipeline_no_rag.process_text("Error code P0420 detected on ECU")
    assert out["command"] == "diagnose_error"
    assert out["parameters"].get("code") == "P0420"


def test_dtc_extraction_p0171(pipeline_no_rag):
    out = pipeline_no_rag.process_text("DTC P0171 fuel system too lean bank 1")
    assert out["command"] == "diagnose_error"
    assert out["parameters"].get("code") == "P0171"


def test_tpms_alert(pipeline_no_rag):
    out = pipeline_no_rag.process_text("TPMS alert front right tire 28 PSI")
    assert out["command"] == "tpms_alert"


def test_abs_warning(pipeline_no_rag):
    out = pipeline_no_rag.process_text("ABS warning light on")
    assert out["command"] == "abs_warning"


# ── Confidence and NLU ───────────────────────────────────────────────────────

def test_confidence_nonzero(pipeline_no_rag):
    out = pipeline_no_rag.process_text("Set temperature to 72")
    assert out["confidence"] > 0.0


def test_nlu_label_present(pipeline_no_rag):
    out = pipeline_no_rag.process_text("Play some music")
    assert out["nlu"].get("label") not in (None, "")


# ── RAG context ──────────────────────────────────────────────────────────────

def test_rag_context_populated(pipeline):
    out = pipeline.process_text("Set temperature to 72")
    assert out["context"].get("success") is True


# ── OOD / unknown ────────────────────────────────────────────────────────────

def test_ood_gives_unknown_command(pipeline_no_rag):
    out = pipeline_no_rag.process_text("Calculate the square root of 144")
    assert out["command"] == "unknown"
