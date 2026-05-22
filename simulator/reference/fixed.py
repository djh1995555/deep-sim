from dataclasses import asdict
from typing import Any, Dict

from simulator.controller.base import ControllerReference
from simulator.vehicle_model.scenario import ScenarioConfig
from simulator.vehicle_model.state import TeacherState


class FixedReferenceProvider:
    def __init__(self, reference: ControllerReference) -> None:
        self.reference = reference

    def reset(self, scenario: ScenarioConfig) -> None:
        pass

    def query(self, t: float, state: TeacherState) -> ControllerReference:
        return self.reference

    def describe(self) -> Dict[str, Any]:
        return {
            "type": "fixed",
            "reference": asdict(self.reference),
        }
