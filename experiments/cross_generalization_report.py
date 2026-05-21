import json
import os
from typing import Any, Dict, List


RUNS = [
    ("R034", "Base", "E1+T1+F1+S1+M1a+V1+U0", "runs/R034_cross_generalization_base"),
    (
        "R035",
        "Final single",
        "E2+T1+F1+S1+M0-fixed+V2-small+U0",
        "runs/R035_cross_generalization_selected_single",
    ),
    (
        "R036",
        "Final + U1",
        "E2+T1+F1+S1+M0-fixed+V2-small+U1",
        "runs/R036_cross_generalization_selected_ensemble",
    ),
]


def main() -> int:
    rows = []
    for run_id, label, variant_label, run_dir in RUNS:
        report_path = os.path.join(run_dir, "artifacts", "base_hybrid_report.json")
        summary_path = os.path.join(run_dir, "summary.json")
        if not os.path.exists(report_path) or not os.path.exists(summary_path):
            rows.append(_missing_row(run_id, label, variant_label))
            continue
        with open(report_path, "r", encoding="utf-8") as handle:
            report = json.load(handle)
        with open(summary_path, "r", encoding="utf-8") as handle:
            summary = json.load(handle)
        metrics = report.get("summary_metrics", {})
        rows.append(
            {
                "run_id": run_id,
                "label": label,
                "variant": variant_label,
                "status": summary.get("status", "missing"),
                "passed": int(bool(summary.get("success_criteria_met"))),
                "claim_supported": int(
                    metrics.get("cross_generalization_claim_supported", 0)
                ),
                "seen_ratio": metrics.get(
                    "seen_config_combined_base_vs_physics_ratio_h100_mean", 0.0
                ),
                "held_out_road_ratio": metrics.get(
                    "held_out_road_base_vs_physics_ratio_h100_mean", 0.0
                ),
                "held_out_vehicle_ratio": metrics.get(
                    "held_out_vehicle_base_vs_physics_ratio_h100_mean", 0.0
                ),
                "held_out_vehicle_vs_black_box": metrics.get(
                    "held_out_vehicle_base_vs_black_box_ratio_h100_mean", 0.0
                ),
                "gap_vehicle_over_seen": metrics.get(
                    "cross_generalization_gap_vehicle_over_seen", 0.0
                ),
                "constraint_rate": metrics.get(
                    "base_constraint_violation_rate_mean", 0.0
                ),
                "uncertainty_ood_lift": metrics.get("uncertainty_ood_lift_proxy", 0.0),
            }
        )
    payload = {"rows": rows, "best_deployable": _best_deployable(rows)}
    _write_json("reports/B5_cross_generalization_summary.json", payload)
    _write_markdown("reports/B5_cross_generalization.md", payload)
    print("wrote reports/B5_cross_generalization.md")
    return 0


def _missing_row(run_id: str, label: str, variant: str) -> Dict[str, Any]:
    return {
        "run_id": run_id,
        "label": label,
        "variant": variant,
        "status": "missing",
        "passed": 0,
        "claim_supported": 0,
        "seen_ratio": 0.0,
        "held_out_road_ratio": 0.0,
        "held_out_vehicle_ratio": 0.0,
        "held_out_vehicle_vs_black_box": 0.0,
        "gap_vehicle_over_seen": 0.0,
        "constraint_rate": 0.0,
        "uncertainty_ood_lift": 0.0,
    }


def _best_deployable(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    deployable = [
        row
        for row in rows
        if row["passed"] and row["label"] in ["Base", "Final single"]
    ]
    if not deployable:
        return {}
    return min(deployable, key=lambda row: row["held_out_vehicle_ratio"])


def _write_json(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def _write_markdown(path: str, payload: Dict[str, Any]) -> None:
    rows = payload["rows"]
    lines = [
        "# B5 Cross-Vehicle / Cross-Config Scaffold Report",
        "",
        "This report aggregates M7 scaffold runs R034-R036. Ratios are 100-step RMSE relative to physics-only unless stated otherwise; lower is better.",
        "",
        "| Run | System | Variant | Passed | Claim Supported | Seen / Phys | Held-out Road / Phys | Held-out Vehicle / Phys | Held-out Vehicle / Black-box | Vehicle Gap | Constraint Rate | Uncertainty OOD Lift |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {run_id} | {label} | {variant} | {passed} | {claim_supported} | {seen_ratio:.6f} | {held_out_road_ratio:.6f} | {held_out_vehicle_ratio:.6f} | {held_out_vehicle_vs_black_box:.6f} | {gap_vehicle_over_seen:.6f} | {constraint_rate:.6f} | {uncertainty_ood_lift:.6f} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "## Selected Deployable Scaffold",
            "",
            "```json",
            json.dumps(payload.get("best_deployable", {}), indent=2, sort_keys=True),
            "```",
            "",
            "## Notes",
            "",
            "- `Final single` keeps `U0` because it is the single-model checkpoint candidate.",
            "- `Final + U1` is the optional K=3 uncertainty wrapper and is not counted as the single-model checkpoint.",
            "- `Claim Supported = 0` means this scaffold run completed but does not yet prove superiority over the black-box held-out-vehicle baseline.",
            "- These are scaffold comparisons over generated DS1 data, not final PyTorch training results.",
            "",
        ]
    )
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


if __name__ == "__main__":
    raise SystemExit(main())
