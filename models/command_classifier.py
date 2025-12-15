"""Command Classification Module.

Classifies natural language to executable vehicle commands.
"""

import logging
from typing import Dict, Any, List, Optional
from enum import Enum

logger = logging.getLogger(__name__)


class VehicleCommand(Enum):
    """Available vehicle commands."""
    START_ENGINE = "start_engine"
    STOP_ENGINE = "stop_engine"
    SET_TEMPERATURE = "set_temperature"
    OPEN_WINDOW = "open_window"
    CLOSE_WINDOW = "close_window"
    LOCK_DOORS = "lock_doors"
    UNLOCK_DOORS = "unlock_doors"
    TURN_ON_LIGHTS = "turn_on_lights"
    TURN_OFF_LIGHTS = "turn_off_lights"
    ADJUST_SEAT = "adjust_seat"
    PLAY_MUSIC = "play_music"
    STOP_MUSIC = "stop_music"
    NAVIGATE = "navigate"
    CALL = "call"
    UNKNOWN = "unknown"


class CommandClassifier:
    """Classifies intent to vehicle commands.
    
    Uses rules and ML to map user intents to actionable vehicle commands.
    """
    
    # Intent to command mappings
    INTENT_COMMAND_MAP = {
        "start": VehicleCommand.START_ENGINE,
        "stop": VehicleCommand.STOP_ENGINE,
        "temperature": VehicleCommand.SET_TEMPERATURE,
        "warm": VehicleCommand.SET_TEMPERATURE,
        "cool": VehicleCommand.SET_TEMPERATURE,
        "window": VehicleCommand.OPEN_WINDOW,
        "open": VehicleCommand.OPEN_WINDOW,
        "close": VehicleCommand.CLOSE_WINDOW,
        "lock": VehicleCommand.LOCK_DOORS,
        "unlock": VehicleCommand.UNLOCK_DOORS,
        "lights": VehicleCommand.TURN_ON_LIGHTS,
        "light": VehicleCommand.TURN_ON_LIGHTS,
        "music": VehicleCommand.PLAY_MUSIC,
        "navigate": VehicleCommand.NAVIGATE,
        "directions": VehicleCommand.NAVIGATE,
        "call": VehicleCommand.CALL,
        "phone": VehicleCommand.CALL,
    }
    
    def __init__(self):
        """Initialize command classifier."""
        self.confidence_threshold = 0.5
        logger.info("CommandClassifier initialized")
    
    def classify(self, intent_label: str, confidence: float, text: str) -> Dict[str, Any]:
        """Classify intent to vehicle command.
        
        Args:
            intent_label: Intent classification result
            confidence: Confidence score of intent classification
            text: Original user text
            
        Returns:
            Dictionary with command, parameters, and confidence
        """
        try:
            command, params = self._map_intent_to_command(intent_label, text)
            return {
                "command": command.value if command else "unknown",
                "parameters": params,
                "confidence": confidence,
                "raw_intent": intent_label
            }
        except Exception as e:
            logger.error(f"Error classifying command: {e}")
            return {
                "command": VehicleCommand.UNKNOWN.value,
                "parameters": {},
                "confidence": 0.0,
                "error": str(e)
            }
    
    def _map_intent_to_command(self, intent_label: str, text: str) -> tuple[VehicleCommand, Dict[str, Any]]:
        """Map intent to vehicle command and extract parameters.
        
        Args:
            intent_label: Intent classification label
            text: Original user text
            
        Returns:
            Tuple of (VehicleCommand, parameters dict)
        """
        text_lower = text.lower()
        
        # Rule-based mapping
        for keyword, command in self.INTENT_COMMAND_MAP.items():
            if keyword in text_lower:
                params = self._extract_parameters(command, text)
                return command, params
        
        return VehicleCommand.UNKNOWN, {}
    
    def _extract_parameters(self, command: VehicleCommand, text: str) -> Dict[str, Any]:
        """Extract command-specific parameters from text.
        
        Args:
            command: The identified command
            text: Original user text
            
        Returns:
            Dictionary of command parameters
        """
        params = {}
        text_lower = text.lower()
        
        if command == VehicleCommand.SET_TEMPERATURE:
            # Extract temperature value
            import re
            temps = re.findall(r'(\d+)\s*(?:degrees?|°|f|c)', text_lower)
            if temps:
                params["temperature"] = int(temps[0])
        
        elif command == VehicleCommand.NAVIGATE:
            # Extract destination
            if "to" in text_lower:
                idx = text_lower.find("to")
                params["destination"] = text[idx+2:].strip()
        
        elif command == VehicleCommand.CALL:
            # Extract contact name
            if "call" in text_lower:
                idx = text_lower.find("call")
                params["contact"] = text[idx+4:].strip()
        
        return params
