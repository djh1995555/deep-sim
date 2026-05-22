from typing import Any, Dict, Optional

from simulator.controller.base import ControllerReference
from simulator.reference.base import ReferenceProvider
from simulator.reference.double_lane_change import (
    DoubleLaneChangeReferenceConfig,
    DoubleLaneChangeReferenceProvider,
)
from simulator.reference.fixed import FixedReferenceProvider
from simulator.reference.lane_change import (
    LaneChangeReferenceConfig,
    LaneChangeReferenceProvider,
)
from simulator.reference.waypoints import (
    WaypointReferenceConfig,
    WaypointReferenceProvider,
    parse_waypoint,
)


def build_reference_provider(
    config: Optional[Dict[str, Any]],
    fallback: ControllerReference,
) -> ReferenceProvider:
    if not config:
        return FixedReferenceProvider(fallback)
    ref_type = str(config.get("type", "fixed"))
    if ref_type == "fixed":
        return FixedReferenceProvider(_fixed_reference(config, fallback))
    if ref_type == "lane_change":
        return LaneChangeReferenceProvider(
            LaneChangeReferenceConfig(
                speed_mps=float(config.get("speed_mps", fallback.target_speed_mps)),
                start_y_m=float(config.get("start_y_m", fallback.target_y_m)),
                end_y_m=float(config.get("end_y_m", 3.5)),
                start_x_m=float(config.get("start_x_m", 20.0)),
                length_m=float(config.get("length_m", 35.0)),
                start_time_s=float(config.get("start_time_s", 0.0)),
                duration_s=float(config.get("duration_s", 3.0)),
                mode=str(config.get("mode", "spatial")),
            )
        )
    if ref_type == "double_lane_change":
        return DoubleLaneChangeReferenceProvider(
            DoubleLaneChangeReferenceConfig(
                speed_mps=float(config.get("speed_mps", fallback.target_speed_mps)),
                start_y_m=float(config.get("start_y_m", fallback.target_y_m)),
                offset_y_m=float(config.get("offset_y_m", 3.5)),
                end_y_m=float(config.get("end_y_m", fallback.target_y_m)),
                start_x_m=float(config.get("start_x_m", 5.0)),
                first_length_m=float(config.get("first_length_m", 18.0)),
                hold_length_m=float(config.get("hold_length_m", 8.0)),
                second_length_m=float(config.get("second_length_m", 18.0)),
                start_time_s=float(config.get("start_time_s", 0.0)),
                mode=str(config.get("mode", "spatial")),
            )
        )
    if ref_type == "waypoints":
        waypoints = [
            parse_waypoint(item)
            for item in config.get("points", config.get("waypoints", []))
        ]
        return WaypointReferenceProvider(
            WaypointReferenceConfig(
                waypoints=waypoints,
                lookahead_m=float(
                    config.get("lookahead_m", fallback.lookahead_distance_m or 8.0)
                ),
            )
        )
    raise ValueError("unsupported reference type: %s" % ref_type)


def _fixed_reference(
    config: Dict[str, Any],
    fallback: ControllerReference,
) -> ControllerReference:
    return ControllerReference(
        target_x_m=float(config.get("x_m", fallback.target_x_m)),
        target_speed_mps=float(config.get("speed_mps", fallback.target_speed_mps)),
        target_y_m=float(config.get("y_m", fallback.target_y_m)),
        target_yaw_rad=float(config.get("yaw_rad", fallback.target_yaw_rad)),
        target_yaw_rate_rps=float(
            config.get("yaw_rate_rps", fallback.target_yaw_rate_rps)
        ),
        target_curvature_1pm=float(
            config.get("curvature_1pm", fallback.target_curvature_1pm)
        ),
        path_s_m=float(config.get("path_s_m", fallback.path_s_m)),
        lookahead_distance_m=float(
            config.get("lookahead_distance_m", fallback.lookahead_distance_m)
        ),
        extra=dict(config.get("extra", fallback.extra)),
    )
