import argparse
import json
import os
from typing import Any, Dict

import yaml

from simulator.visualizer.debug_trace import DebugTrace


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    panels = _load_panels(args.panels)
    output_path = args.out or os.path.join(
        os.path.dirname(os.path.abspath(args.trace)),
        "debug_report.html",
    )
    trace = DebugTrace.read_json(args.trace)
    trace.write_html(output_path, panels=panels, title=args.title)
    print(
        json.dumps(
            {
                "status": "success",
                "trace": args.trace,
                "out": output_path,
                "sample_count": len(trace.to_list()),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a Plotly HTML report from simulator debug_trace.json."
    )
    parser.add_argument("--trace", required=True, help="path to debug_trace.json")
    parser.add_argument("--out", help="output HTML path; defaults next to trace")
    parser.add_argument(
        "--panels",
        help="optional YAML/JSON mapping of panel names to debug trace signal names",
    )
    parser.add_argument("--title", default="Simulator Debug Report")
    return parser


def _load_panels(path: str | None) -> Dict[str, Any] | None:
    if not path:
        return None
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError("panels config must be a mapping")
    return data


if __name__ == "__main__":
    raise SystemExit(main())
