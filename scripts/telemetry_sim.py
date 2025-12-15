"""Simulate telemetry and show how context can be incorporated.

This script prints mock telemetry and demonstrates retrieving context from VehicleRAG.
"""

from __future__ import annotations

import json

from models.rag_component import VehicleRAG
from models.telemetry import get_mock_telemetry


def main():
    rag = VehicleRAG()
    telemetry = get_mock_telemetry()
    print(json.dumps({"telemetry": telemetry, "vehicle_status": rag.get_vehicle_status()}, indent=2))


if __name__ == "__main__":
    main()
