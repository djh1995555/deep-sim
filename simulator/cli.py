import argparse
import json
from dataclasses import fields
from typing import Any, Dict, Tuple

from simulator.simulator_app import (
    ClosedLoopSimulationRequest,
    load_simulation_request,
    run_closed_loop_simulation,
)


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    try:
        request = (
            load_simulation_request(args.request)
            if args.request
            else ClosedLoopSimulationRequest()
        )
        request_overrides, model_overrides, scenario_overrides = _cli_overrides(args)
        request = request.with_overrides(request_overrides)
        if model_overrides:
            request = request.with_overrides(
                {"model": {**request.model, **model_overrides}}
            )
        if scenario_overrides:
            request = request.with_overrides(
                {"scenario": _merge_dict(request.scenario, scenario_overrides)}
            )
        summary = run_closed_loop_simulation(request)
    except ValueError as exc:
        print("configuration error: %s" % exc)
        return 2
    except FloatingPointError as exc:
        print("numerical integration failure: %s" % exc)
        return 3
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a closed-loop vehicle simulation episode."
    )
    parser.add_argument("--request", help="optional YAML request file")
    parser.add_argument("--out-dir")
    parser.add_argument("--scenario-id")
    parser.add_argument("--road")
    parser.add_argument("--vehicle-index", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--initial-speed-mps", type=float)
    parser.add_argument("--duration-s", type=float)
    parser.add_argument("--dt-internal", type=float)
    parser.add_argument("--dt-export", type=float)
    parser.add_argument(
        "--reference-json",
        help="JSON reference provider config, for example '{\"type\":\"lane_change\"}'",
    )
    parser.add_argument("--reference-file", help="YAML reference config file")
    parser.add_argument("--pid-kp", type=float)
    parser.add_argument("--pid-ki", type=float)
    parser.add_argument("--pid-kd", type=float)
    parser.add_argument("--lqr-gains", type=float, nargs=4)
    parser.add_argument("--lqr-max-sw-angle-rad", type=float)
    parser.add_argument("--controller-type")
    parser.add_argument(
        "--mpc-config-json",
        help="JSON coupled MPC config, for example '{\"horizon_steps\":8}'",
    )
    parser.add_argument("--debug-stride", type=int)
    parser.add_argument("--write-debug-trace", action=argparse.BooleanOptionalAction)
    parser.add_argument("--write-debug-html", action=argparse.BooleanOptionalAction)
    parser.add_argument("--timestamped-output", action=argparse.BooleanOptionalAction)
    return parser


def _cli_overrides(
    args: argparse.Namespace,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    valid = {field.name for field in fields(ClosedLoopSimulationRequest)}
    overrides: Dict[str, Any] = {}
    model_overrides: Dict[str, Any] = {}
    scenario_overrides: Dict[str, Any] = {}
    model_keys = {"duration_s", "dt_internal", "dt_export"}
    for key, value in vars(args).items():
        if key == "request" or value is None:
            continue
        if key == "reference_json":
            scenario_overrides["reference"] = json.loads(value)
            continue
        if key == "reference_file":
            scenario_overrides["reference"] = value
            continue
        if key == "mpc_config_json":
            overrides["mpc_config"] = json.loads(value)
            continue
        if key in model_keys:
            model_overrides[key] = value
            continue
        if key == "scenario_id":
            scenario_overrides["id"] = value
            continue
        if key == "road":
            scenario_overrides["road"] = value
            continue
        if key == "vehicle_index":
            scenario_overrides["vehicle_index"] = value
            continue
        if key == "seed":
            scenario_overrides["seed"] = value
            continue
        if key == "initial_speed_mps":
            scenario_overrides.setdefault("initial_state", {})["speed_mps"] = value
            continue
        if key not in valid:
            raise ValueError("unsupported CLI option maps to field '%s'" % key)
        overrides[key] = value
    return overrides, model_overrides, scenario_overrides


def _merge_dict(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge_dict(result[key], value)
        else:
            result[key] = value
    return result


if __name__ == "__main__":
    raise SystemExit(main())
