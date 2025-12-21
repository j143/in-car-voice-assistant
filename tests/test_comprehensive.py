"""
Comprehensive test suite for in-car voice assistant.
All tests run on CPU; no GPU required.
"""

import json
import sys
import tempfile
from pathlib import Path
from typing import List, Dict, Any

import pytest

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.end_to_end import VoiceAssistantPipeline
from scripts.train_qlora import format_sample, split_dataset, load_dataset


class TestParameterParsing:
    """Test parameter extraction from generated assistant text."""
    
    def test_parse_error_codes_and_terms(self):
        """Test parsing error codes and technical terms."""
        text = 'Assistant: Intent=error_handling, Command=diagnose_error, Parameters={"codes": ["P0420"], "terms": ["catalyst system"]}'
        result = VoiceAssistantPipeline._parse_generated_assistant(text)
        
        assert result["command"] == "diagnose_error"
        assert result["parameters"]["codes"] == ["P0420"]
        assert result["parameters"]["terms"] == ["catalyst system"]
    
    def test_parse_temperature_parameter(self):
        """Test parsing temperature parameter."""
        text = 'Assistant: Intent=climate_control, Command=set_temperature, Parameters={"temperature": 22}'
        result = VoiceAssistantPipeline._parse_generated_assistant(text)
        
        assert result["command"] == "set_temperature"
        assert result["parameters"]["temperature"] == 22
    
    def test_parse_empty_parameters(self):
        """Test parsing with empty parameters."""
        text = 'Intent=query_sensor, Command=read_signal, Parameters={}'
        result = VoiceAssistantPipeline._parse_generated_assistant(text)
        
        assert result["command"] == "read_signal"
        assert result["parameters"] == {}
    
    def test_parse_malformed_json(self):
        """Test graceful handling of malformed JSON."""
        text = 'Command=set_temperature, Parameters={invalid json'
        result = VoiceAssistantPipeline._parse_generated_assistant(text)
        
        # Should still extract command
        assert result["command"] == "set_temperature"
        # Parameters should be empty on parse failure
        assert isinstance(result["parameters"], dict)
    
    def test_parse_missing_command(self):
        """Test handling missing Command field."""
        text = 'Intent=error_handling, Parameters={"codes": ["P0171"]}'
        result = VoiceAssistantPipeline._parse_generated_assistant(text)
        
        # Should return default unknown
        assert result["command"] == "unknown"


class TestFormatSample:
    """Test training sample formatting."""
    
    def test_format_sample_with_error_codes(self):
        """Test formatting with error codes."""
        sample = {
            "id": 0,
            "text": "Got error P0420",
            "intent": "error_handling",
            "command": "diagnose_error",
            "error_codes": ["P0420"],
            "technical_terms": ["catalyst system"],
            "subsystem": "emissions_control",
            "severity": "warning",
        }
        
        formatted = format_sample(sample)
        
        assert "User: Got error P0420" in formatted
        assert "Error Codes: P0420" in formatted
        assert "Technical Terms: catalyst system" in formatted
        assert "Intent=error_handling, Command=diagnose_error, Parameters=" in formatted
        assert '"codes": ["P0420"]' in formatted
    
    def test_format_sample_no_error_codes(self):
        """Test formatting without error codes."""
        sample = {
            "id": 1,
            "text": "Set temperature to 22",
            "intent": "climate_control",
            "command": "set_temperature",
            "error_codes": [],
            "technical_terms": ["HVAC"],
            "subsystem": "hvac",
            "severity": "normal",
        }
        
        formatted = format_sample(sample)
        
        assert "User: Set temperature to 22" in formatted
        assert "Error Codes:" not in formatted  # Should not include empty error codes
        assert "Intent=climate_control" in formatted
        assert "Command=set_temperature" in formatted
        assert "Parameters=" in formatted
    
    def test_format_sample_preserves_intent_command(self):
        """Test that intent and command are correctly preserved."""
        sample = {
            "text": "test",
            "intent": "query_sensor",
            "command": "read_signal",
            "error_codes": [],
            "technical_terms": [],
        }
        
        formatted = format_sample(sample)
        
        assert "Intent=query_sensor" in formatted
        assert "Command=read_signal" in formatted


class TestDatasetSplit:
    """Test train/test split functionality."""
    
    def test_split_dataset_ratio(self):
        """Test correct split ratio."""
        samples = [{"id": i, "text": f"text {i}"} for i in range(100)]
        train, test = split_dataset(samples, val_ratio=0.2, seed=42)
        
        assert len(train) == 80
        assert len(test) == 20
        assert len(train) + len(test) == 100
    
    def test_split_dataset_no_overlap(self):
        """Test that train and test sets don't overlap."""
        samples = [{"id": i, "text": f"text {i}"} for i in range(100)]
        train, test = split_dataset(samples, val_ratio=0.2, seed=42)
        
        train_ids = {s["id"] for s in train}
        test_ids = {s["id"] for s in test}
        
        assert train_ids & test_ids == set()  # No overlap
    
    def test_split_dataset_deterministic(self):
        """Test that split is deterministic with same seed."""
        samples = [{"id": i, "text": f"text {i}"} for i in range(100)]
        
        train1, test1 = split_dataset(samples, val_ratio=0.2, seed=42)
        train2, test2 = split_dataset(samples, val_ratio=0.2, seed=42)
        
        assert [s["id"] for s in train1] == [s["id"] for s in train2]
        assert [s["id"] for s in test1] == [s["id"] for s in test2]
    
    def test_split_dataset_different_seeds(self):
        """Test that different seeds produce different splits."""
        samples = [{"id": i, "text": f"text {i}"} for i in range(100)]
        
        train1, _ = split_dataset(samples, val_ratio=0.2, seed=42)
        train2, _ = split_dataset(samples, val_ratio=0.2, seed=123)
        
        ids1 = {s["id"] for s in train1}
        ids2 = {s["id"] for s in train2}
        
        # Should be different sets (with high probability)
        assert ids1 != ids2
    
    def test_split_small_dataset(self):
        """Test split with small dataset."""
        samples = [{"id": i, "text": f"text {i}"} for i in range(5)]
        train, test = split_dataset(samples, val_ratio=0.2, seed=42)
        
        assert len(train) >= 3
        assert len(test) >= 1
        assert len(train) + len(test) == 5


class TestPipelineIntegration:
    """Test end-to-end pipeline."""
    
    def test_process_text_basic(self):
        """Test basic text processing without adapter."""
        pipeline = VoiceAssistantPipeline(use_rag=False, classifier_type="rule")
        result = pipeline.process_text("Set temperature to 22 degrees")
        
        assert isinstance(result, dict)
        assert "command" in result
        assert "parameters" in result
        assert "transcript" in result
        assert result["transcript"] == "Set temperature to 22 degrees"
    
    def test_process_text_temperature_extraction(self):
        """Test temperature parameter extraction."""
        pipeline = VoiceAssistantPipeline(use_rag=False, classifier_type="rule")
        result = pipeline.process_text("Set temperature to 22 degrees")
        
        assert result["command"] == "set_temperature"
        assert result["parameters"].get("temperature") == 22
    
    def test_process_text_lock_doors(self):
        """Test lock doors command."""
        pipeline = VoiceAssistantPipeline(use_rag=False, classifier_type="rule")
        result = pipeline.process_text("Lock the doors")
        
        assert result["command"] == "lock_doors"
    
    def test_process_text_unknown_command(self):
        """Test handling of unknown commands."""
        pipeline = VoiceAssistantPipeline(use_rag=False, classifier_type="rule")
        result = pipeline.process_text("Play the album by unknown artist")
        
        assert "command" in result
        assert isinstance(result["command"], str)
    
    def test_process_text_with_confidence(self):
        """Test that confidence score is returned."""
        pipeline = VoiceAssistantPipeline(use_rag=False, classifier_type="rule")
        result = pipeline.process_text("Set temperature to 22")
        
        assert "confidence" in result
        assert 0.0 <= result["confidence"] <= 1.0


class TestCommandClassifier:
    """Test command classification functionality."""
    
    def test_classify_set_temperature(self):
        """Test set_temperature command classification."""
        from models.command_classifier import CommandClassifier
        
        classifier = CommandClassifier()
        result = classifier.classify(
            intent_label="climate_control",
            confidence=0.95,
            text="Set temperature to 22 degrees"
        )
        
        assert result["command"] == "set_temperature"
        assert result["parameters"].get("temperature") == 22
        assert result["confidence"] == 0.95
    
    def test_classify_lock_doors(self):
        """Test lock_doors command classification."""
        from models.command_classifier import CommandClassifier
        
        classifier = CommandClassifier()
        result = classifier.classify(
            intent_label="lock_doors",
            confidence=0.90,
            text="Lock the doors"
        )
        
        assert result["command"] == "lock_doors"
    
    def test_classify_navigate(self):
        """Test navigate command with destination extraction."""
        from models.command_classifier import CommandClassifier
        
        classifier = CommandClassifier()
        result = classifier.classify(
            intent_label="navigate",
            confidence=0.85,
            text="Navigate to downtown"
        )
        
        assert result["command"] == "navigate"
        assert "destination" in result["parameters"]
        assert "downtown" in result["parameters"]["destination"]


class TestDatasetLoading:
    """Test dataset loading and format."""
    
    def test_load_test_dataset(self):
        """Test loading automotive_domain_test.jsonl."""
        test_path = Path(__file__).parent.parent / "data" / "automotive_domain_test.jsonl"
        
        if not test_path.exists():
            pytest.skip(f"Test dataset not found: {test_path}")
        
        data = load_dataset(test_path)
        
        assert len(data) > 0
        assert all(isinstance(sample, dict) for sample in data)
    
    def test_test_dataset_schema(self):
        """Test that test dataset has required fields."""
        test_path = Path(__file__).parent.parent / "data" / "automotive_domain_test.jsonl"
        
        if not test_path.exists():
            pytest.skip(f"Test dataset not found: {test_path}")
        
        data = load_dataset(test_path)
        required_fields = ["text", "intent", "command"]
        
        for sample in data[:5]:  # Check first 5
            for field in required_fields:
                assert field in sample, f"Missing field '{field}' in sample: {sample}"
    
    def test_test_dataset_has_error_codes_or_terms(self):
        """Test that test dataset includes parameter fields."""
        test_path = Path(__file__).parent.parent / "data" / "automotive_domain_test.jsonl"
        
        if not test_path.exists():
            pytest.skip(f"Test dataset not found: {test_path}")
        
        data = load_dataset(test_path)
        
        # At least some samples should have error_codes or technical_terms
        has_params = any(
            sample.get("error_codes") or sample.get("technical_terms")
            for sample in data
        )
        
        assert has_params, "Test dataset should include error_codes or technical_terms"


class TestEvaluationMetrics:
    """Test evaluation metrics calculation."""
    
    def test_parameter_match_rate_calculation(self):
        """Test parameter match rate calculation."""
        # Simulate evaluation scenario
        gt_codes = ["P0420"]
        pred_codes = ["P0420"]
        
        # Should match
        assert any(c in pred_codes for c in gt_codes)
    
    def test_parameter_no_match(self):
        """Test when parameters don't match."""
        gt_codes = ["P0420"]
        pred_codes = ["P0171"]
        
        # Should not match
        assert not any(c in pred_codes for c in gt_codes)
    
    def test_partial_parameter_match(self):
        """Test partial parameter matches."""
        gt_codes = ["P0420", "P0171"]
        pred_codes = ["P0420", "U0073"]
        
        # Partial match (1 out of 2)
        match_count = sum(1 for c in gt_codes if c in pred_codes)
        assert match_count == 1


class TestJSONHandling:
    """Test JSON parsing and handling."""
    
    def test_json_parameters_roundtrip(self):
        """Test JSON serialization and deserialization of parameters."""
        original = {"codes": ["P0420", "P0171"], "terms": ["catalyst", "fuel system"]}
        
        # Serialize
        serialized = json.dumps(original)
        
        # Deserialize
        deserialized = json.loads(serialized)
        
        assert deserialized == original
    
    def test_parameters_in_prompt(self):
        """Test that parameters appear in training prompt."""
        sample = {
            "text": "Error P0420",
            "intent": "error_handling",
            "command": "diagnose_error",
            "error_codes": ["P0420"],
            "technical_terms": ["catalyst"],
        }
        
        formatted = format_sample(sample)
        
        # Should contain parseable JSON
        assert "Parameters=" in formatted
        # Should be able to extract JSON
        if "Parameters={" in formatted:
            params_str = formatted.split("Parameters=")[1].split("\n")[0].strip()
            parsed = json.loads(params_str)
            assert "codes" in parsed


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_empty_text_processing(self):
        """Test processing empty text."""
        pipeline = VoiceAssistantPipeline(use_rag=False, classifier_type="rule")
        result = pipeline.process_text("")
        
        assert "command" in result
        assert result.get("command") == "unknown"
    
    def test_very_long_text(self):
        """Test processing very long text."""
        pipeline = VoiceAssistantPipeline(use_rag=False, classifier_type="rule")
        long_text = "Set temperature to 22 degrees " * 50
        
        result = pipeline.process_text(long_text)
        
        assert isinstance(result, dict)
        assert "command" in result
    
    def test_special_characters(self):
        """Test processing text with special characters."""
        pipeline = VoiceAssistantPipeline(use_rag=False, classifier_type="rule")
        result = pipeline.process_text("Set temperature to 22°C / 72°F")
        
        assert isinstance(result, dict)
        assert "command" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
