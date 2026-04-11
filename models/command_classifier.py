"""Command Classification Module.

Classifies natural language to executable vehicle commands.
"""

import logging
import re
from typing import Any, Dict, Optional, Tuple
from enum import Enum

logger = logging.getLogger(__name__)

# OBD-II / DTC error code pattern (e.g. P0420, B1234, U0100)
_DTC_PATTERN = re.compile(r'\b([PBCU]\d{4})\b', re.I)
# Mercedes / Bosch part-number pattern (e.g. K 300 503 v17, A 000 420 17 20)
_PART_PATTERN = re.compile(r'\b([A-Z]\s?\d{3}\s?\d{3}(?:\s?\w+)?)\b', re.I)
# Bare temperature number: "to 72", "at 22", or plain "72"
_TEMP_PATTERN = re.compile(
    r'(?:to|at|=)?\s*(\d{1,3})\s*(?:degrees?|°|fahrenheit|celsius|°[fc]|[fc]\b)?',
    re.I,
)
# Percentage (volume, fan speed)
_PERCENT_PATTERN = re.compile(r'(\d{1,3})\s*(?:percent|%)')
# Ordinal / level number
_LEVEL_PATTERN = re.compile(r'(?:level|to)\s+(\d+)')


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
    SET_SEAT_HEATING = "set_seat_heating"
    PLAY_MUSIC = "play_music"
    PAUSE_MUSIC = "pause_music"
    NEXT_TRACK = "next_track"
    SET_VOLUME = "set_volume"
    NAVIGATE = "navigate"
    SHOW_TRAFFIC = "show_traffic"
    SHOW_ETA = "show_eta"
    CALL = "call"
    DIAGNOSE_ERROR = "diagnose_error"
    BATTERY_WARNING = "battery_warning"
    TPMS_ALERT = "tpms_alert"
    ABS_WARNING = "abs_warning"
    ESC_WARNING = "esc_warning"
    BRAKE_WARNING = "brake_warning"
    COOLANT_WARNING = "coolant_warning"
    DEFROST = "defrost"
    SET_FAN_SPEED = "set_fan_speed"
    ADJUST_CLIMATE = "adjust_climate"
    READ_SIGNAL = "read_signal"
    HISTORICAL_QUERY = "historical_query"
    SET_ALERT = "set_alert"
    UNKNOWN = "unknown"


class CommandClassifier:
    """Classifies natural-language text to executable vehicle commands.

    Maps intent labels *and* text keywords to a VehicleCommand, then
    extracts command-specific parameters.
    """

    # Intent label → default command when text keywords don't fire first
    INTENT_COMMAND_MAP: Dict[str, VehicleCommand] = {
        "climate_control": VehicleCommand.ADJUST_CLIMATE,
        "error_handling": VehicleCommand.DIAGNOSE_ERROR,
        "battery_warning": VehicleCommand.BATTERY_WARNING,
        "music_control": VehicleCommand.PLAY_MUSIC,
        "navigation": VehicleCommand.NAVIGATE,
        "query_sensor": VehicleCommand.READ_SIGNAL,
        "query_historical": VehicleCommand.HISTORICAL_QUERY,
        "set_alert": VehicleCommand.SET_ALERT,
        "ood": VehicleCommand.UNKNOWN,
    }

    # Ordered list of (keyword, command) for text-level matching.
    # More specific phrases first.
    _TEXT_RULES: list = [
        # Error / diagnostics
        ("dtc", VehicleCommand.DIAGNOSE_ERROR),
        ("error code", VehicleCommand.DIAGNOSE_ERROR),
        ("fault code", VehicleCommand.DIAGNOSE_ERROR),
        ("check engine", VehicleCommand.DIAGNOSE_ERROR),
        ("abs warning", VehicleCommand.ABS_WARNING),
        ("esc", VehicleCommand.ESC_WARNING),
        ("tpms", VehicleCommand.TPMS_ALERT),
        ("tire pressure", VehicleCommand.TPMS_ALERT),
        ("brake warning", VehicleCommand.BRAKE_WARNING),
        ("brake pad", VehicleCommand.BRAKE_WARNING),
        ("coolant", VehicleCommand.COOLANT_WARNING),
        ("battery low", VehicleCommand.BATTERY_WARNING),
        ("battery warning", VehicleCommand.BATTERY_WARNING),
        ("battery management", VehicleCommand.BATTERY_WARNING),
        # Climate
        ("defrost", VehicleCommand.DEFROST),
        ("heated seat", VehicleCommand.SET_SEAT_HEATING),
        ("seat heat", VehicleCommand.SET_SEAT_HEATING),
        ("fan speed", VehicleCommand.SET_FAN_SPEED),
        ("temperature", VehicleCommand.SET_TEMPERATURE),
        ("hvac", VehicleCommand.SET_TEMPERATURE),
        ("warm", VehicleCommand.SET_TEMPERATURE),
        ("cool", VehicleCommand.ADJUST_CLIMATE),
        # Music
        ("next track", VehicleCommand.NEXT_TRACK),
        ("pause", VehicleCommand.PAUSE_MUSIC),
        ("volume", VehicleCommand.SET_VOLUME),
        ("play", VehicleCommand.PLAY_MUSIC),
        ("music", VehicleCommand.PLAY_MUSIC),
        ("audio", VehicleCommand.PLAY_MUSIC),
        # Navigation
        ("traffic", VehicleCommand.SHOW_TRAFFIC),
        ("eta", VehicleCommand.SHOW_ETA),
        ("navigate", VehicleCommand.NAVIGATE),
        ("route", VehicleCommand.NAVIGATE),
        ("directions", VehicleCommand.NAVIGATE),
        ("destination", VehicleCommand.NAVIGATE),
        # Windows / doors / lights
        ("unlock", VehicleCommand.UNLOCK_DOORS),
        ("lock", VehicleCommand.LOCK_DOORS),
        ("open", VehicleCommand.OPEN_WINDOW),
        ("close", VehicleCommand.CLOSE_WINDOW),
        ("lights", VehicleCommand.TURN_ON_LIGHTS),
        ("light", VehicleCommand.TURN_ON_LIGHTS),
        # Engine
        ("start engine", VehicleCommand.START_ENGINE),
        ("start the engine", VehicleCommand.START_ENGINE),
        ("stop engine", VehicleCommand.STOP_ENGINE),
        # Sensor query
        ("fuel level", VehicleCommand.READ_SIGNAL),
        ("speed", VehicleCommand.READ_SIGNAL),
        ("rpm", VehicleCommand.READ_SIGNAL),
        # Call
        ("call", VehicleCommand.CALL),
        ("phone", VehicleCommand.CALL),
    ]

    def __init__(self) -> None:
        self.confidence_threshold = 0.5
        logger.info("CommandClassifier initialized")

    def classify(self, intent_label: str, confidence: float, text: str) -> Dict[str, Any]:
        """Return dict with command, parameters, confidence, raw_intent."""
        try:
            command, params = self._map(intent_label, text)
            # Boost confidence to at least 0.7 when we have a solid text match
            effective_confidence = max(confidence, 0.7) if command != VehicleCommand.UNKNOWN else confidence
            return {
                "command": command.value,
                "parameters": params,
                "confidence": effective_confidence,
                "raw_intent": intent_label,
            }
        except Exception as e:
            logger.error("Error classifying command: %s", e)
            return {
                "command": VehicleCommand.UNKNOWN.value,
                "parameters": {},
                "confidence": 0.0,
                "error": str(e),
            }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _map(self, intent_label: str, text: str) -> Tuple[VehicleCommand, Dict[str, Any]]:
        text_lower = text.lower()

        # 1. Text-keyword rules (ordered, most specific first)
        for phrase, cmd in self._TEXT_RULES:
            if phrase in text_lower:
                return cmd, self._extract_parameters(cmd, text)

        # 2. DTC pattern shortcut
        if _DTC_PATTERN.search(text):
            return VehicleCommand.DIAGNOSE_ERROR, self._extract_parameters(VehicleCommand.DIAGNOSE_ERROR, text)

        # 3. Intent-label fallback
        cmd = self.INTENT_COMMAND_MAP.get(intent_label, VehicleCommand.UNKNOWN)
        return cmd, self._extract_parameters(cmd, text)

    def _extract_parameters(self, command: VehicleCommand, text: str) -> Dict[str, Any]:
        params: Dict[str, Any] = {}
        text_lower = text.lower()

        if command == VehicleCommand.SET_TEMPERATURE:
            m = _TEMP_PATTERN.search(text_lower)
            if m:
                params["temperature"] = int(m.group(1))
                # Detect unit
                unit_part = (m.group(0) or "").lower()
                if "celsius" in unit_part or unit_part.endswith("c"):
                    params["unit"] = "Celsius"
                elif "fahrenheit" in unit_part or unit_part.endswith("f"):
                    params["unit"] = "Fahrenheit"

        elif command == VehicleCommand.SET_VOLUME:
            m = _PERCENT_PATTERN.search(text_lower) or _LEVEL_PATTERN.search(text_lower)
            if m:
                params["level"] = int(m.group(1))

        elif command == VehicleCommand.SET_FAN_SPEED:
            m = _LEVEL_PATTERN.search(text_lower) or _PERCENT_PATTERN.search(text_lower)
            if m:
                params["fan_speed"] = int(m.group(1))

        elif command == VehicleCommand.NAVIGATE:
            # Grab text after "to" or "navigate to"
            m = re.search(r'(?:navigate|route|go|directions?)\s+to\s+(.+)', text_lower)
            if not m:
                m = re.search(r'\bto\s+(.+)', text_lower)
            if m:
                params["destination"] = m.group(1).strip()

        elif command == VehicleCommand.DIAGNOSE_ERROR:
            dtc = _DTC_PATTERN.findall(text)
            if dtc:
                params["code"] = dtc[0].upper()
            part = _PART_PATTERN.findall(text)
            if part:
                params["part"] = part[0]

        elif command == VehicleCommand.BATTERY_WARNING:
            part = _PART_PATTERN.findall(text)
            if part:
                params["part"] = part[0]

        elif command in (VehicleCommand.TPMS_ALERT,):
            # Extract tire location
            for loc in ("front left", "front right", "rear left", "rear right",
                        "fl", "fr", "rl", "rr"):
                if loc in text_lower:
                    params["tire"] = loc
                    break
            # Extract pressure reading
            m = re.search(r'(\d+)\s*(?:psi|bar)', text_lower)
            if m:
                params["pressure"] = int(m.group(1))

        elif command == VehicleCommand.CALL:
            m = re.search(r'(?:call|phone)\s+(.+)', text_lower)
            if m:
                params["contact"] = m.group(1).strip()

        elif command == VehicleCommand.SET_SEAT_HEATING:
            if any(w in text_lower for w in ("on", "heat", "warm")):
                params["heat"] = "on"
            elif "off" in text_lower:
                params["heat"] = "off"

        return params
