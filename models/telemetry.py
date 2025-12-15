"""Mock vehicle telemetry adapter.

Provides a simple interface to fetch current telemetry for RAG/context.
"""

from __future__ import annotations

from typing import Dict


def get_mock_telemetry() -> Dict:
    return {
        "speed_kmh": 60,
        "fuel_level_pct": 45,
        "tire_pressure_psi": {"fl": 33, "fr": 33, "rl": 35, "rr": 35},
        "battery_voltage": 12.4,
        "dtc_codes": ["P0420"],
    }
