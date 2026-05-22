from dataclasses import asdict
from typing import Any, Dict, Protocol

from simulator.controller.base import ControllerReference
from simulator.vehicle_model.scenario import ScenarioConfig
from simulator.vehicle_model.state import TeacherState


class ReferenceProvider(Protocol):
    def reset(self, scenario: ScenarioConfig) -> None:
        ...

    def query(self, t: float, state: TeacherState) -> ControllerReference:
        ...

    def describe(self) -> Dict[str, Any]:
        ...


def reference_to_dict(reference: ControllerReference) -> Dict[str, Any]:
    return asdict(reference)
