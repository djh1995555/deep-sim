import csv
import json
import os
from typing import Any, Dict, Iterable, List, Optional


class DebugTrace:
    def __init__(self) -> None:
        self.rows: List[Dict[str, Any]] = []

    def append(
        self,
        t: float,
        controller_input: Dict[str, Any],
        controller_output: Dict[str, Any],
        vehicle_debug: Optional[Dict[str, Any]] = None,
    ) -> None:
        row: Dict[str, Any] = {"t": float(t)}
        row.update(_flatten("input", controller_input))
        row.update(_flatten("output", controller_output))
        if vehicle_debug:
            row.update(_flatten("vehicle", vehicle_debug))
        self.rows.append(row)

    def to_list(self) -> List[Dict[str, Any]]:
        return list(self.rows)

    def write_json(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.rows, handle, indent=2, sort_keys=True)

    def write_csv(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fieldnames = _ordered_fieldnames(self.rows)
        with open(path, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in self.rows:
                writer.writerow({key: _csv_value(row.get(key)) for key in fieldnames})

    def plot_timeseries(self, path: str, signals: Iterable[str]) -> None:
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise RuntimeError("matplotlib is required for plot_timeseries") from exc
        os.makedirs(os.path.dirname(path), exist_ok=True)
        signal_list = list(signals)
        t = [float(row.get("t", 0.0)) for row in self.rows]
        fig, axes = plt.subplots(
            len(signal_list), 1, sharex=True, figsize=(9, 2.2 * len(signal_list))
        )
        if not hasattr(axes, "__iter__"):
            axes = [axes]
        for ax, signal in zip(axes, signal_list):
            ax.plot(t, [row.get(signal, 0.0) for row in self.rows])
            ax.set_ylabel(signal)
            ax.grid(True, alpha=0.25)
        axes[-1].set_xlabel("time [s]")
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)


def _flatten(prefix: str, value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        result: Dict[str, Any] = {}
        for key, child in value.items():
            child_prefix = "%s.%s" % (prefix, key)
            result.update(_flatten(child_prefix, child))
        return result
    if isinstance(value, (list, tuple)):
        return {
            "%s.%d" % (prefix, idx): item
            for idx, item in enumerate(value)
            if _is_scalar(item)
        }
    if _is_scalar(value):
        return {prefix: value}
    return {prefix: str(value)}


def _is_scalar(value: Any) -> bool:
    return isinstance(value, (str, int, float, bool)) or value is None


def _ordered_fieldnames(rows: List[Dict[str, Any]]) -> List[str]:
    keys = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                keys.append(key)
                seen.add(key)
    return keys


def _csv_value(value: Any) -> Any:
    if _is_scalar(value):
        return value
    return json.dumps(value, sort_keys=True)
