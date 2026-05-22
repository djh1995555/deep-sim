import argparse
import json
from dataclasses import fields
from typing import Any, Dict

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
        request = request.with_overrides(_cli_overrides(args))
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
    parser.add_argument("--teacher-config")
    parser.add_argument("--out-dir")
    parser.add_argument("--scenario-id")
    parser.add_argument("--dataset-id")
    parser.add_argument("--road")
    parser.add_argument("--vehicle-index", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--initial-speed-mps", type=float)
    parser.add_argument("--duration-s", type=float)
    parser.add_argument("--dt-internal", type=float)
    parser.add_argument("--dt-export", type=float)
    parser.add_argument("--split-role")
    parser.add_argument("--target-speed-mps", type=float)
    parser.add_argument("--target-y-m", type=float)
    parser.add_argument("--target-yaw-rad", type=float)
    parser.add_argument("--target-yaw-rate-rps", type=float)
    parser.add_argument(
        "--reference-json",
        help="JSON reference provider config, for example '{\"type\":\"lane_change\"}'",
    )
    parser.add_argument("--pid-kp", type=float)
    parser.add_argument("--pid-ki", type=float)
    parser.add_argument("--pid-kd", type=float)
    parser.add_argument("--lqr-gains", type=float, nargs=4)
    parser.add_argument("--lqr-max-sw-angle-rad", type=float)
    parser.add_argument("--debug-stride", type=int)
    parser.add_argument("--write-debug-trace", action=argparse.BooleanOptionalAction)
    return parser


def _cli_overrides(args: argparse.Namespace) -> Dict[str, Any]:
    valid = {field.name for field in fields(ClosedLoopSimulationRequest)}
    overrides: Dict[str, Any] = {}
    for key, value in vars(args).items():
        if key == "request" or value is None:
            continue
        if key == "reference_json":
            overrides["reference"] = json.loads(value)
            continue
        if key not in valid:
            raise ValueError("unsupported CLI option maps to field '%s'" % key)
        overrides[key] = value
    return overrides


if __name__ == "__main__":
    raise SystemExit(main())
