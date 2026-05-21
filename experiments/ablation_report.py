import json
import os
from typing import Any, Dict, List


RUNS = [
    ("R015", "Tire", "T0", "runs/R015_ablation_tire_T0"),
    ("R016", "Tire", "T1", "runs/R016_ablation_tire_T1"),
    ("R017", "Tire", "T1-no-proj", "runs/R017_ablation_tire_T1_no_proj"),
    ("R018", "Tire", "T2", "runs/R018_ablation_tire_T2"),
    ("R019", "Fz", "F0", "runs/R019_ablation_fz_F0"),
    ("R020", "Fz", "F1", "runs/R020_ablation_fz_F1"),
    ("R021", "Fz", "F2", "runs/R021_ablation_fz_F2"),
    ("R022", "Steering", "S0", "runs/R022_ablation_steering_S0"),
    ("R023", "Steering", "S1", "runs/R023_ablation_steering_S1"),
    ("R024", "Mu", "M0-fixed", "runs/R024_ablation_mu_M0_fixed"),
    ("R025", "Mu", "M1a", "runs/R025_ablation_mu_M1a"),
    ("R026", "Mu", "M1b", "runs/R026_ablation_mu_M1b"),
    ("R027", "Mu", "M2-oracle", "runs/R027_ablation_mu_M2_oracle"),
    ("R027a", "Encoder", "E1", "runs/R027a_ablation_encoder_E1"),
    ("R027b", "Encoder", "E2", "runs/R027b_ablation_encoder_E2"),
    ("R027c", "Encoder", "E3", "runs/R027c_ablation_encoder_E3"),
    ("R028", "Vehicle", "V0", "runs/R028_ablation_vehicle_V0"),
    ("R029", "Vehicle", "V1", "runs/R029_ablation_vehicle_V1"),
    ("R030", "Vehicle", "V1-large", "runs/R030_ablation_vehicle_V1_large"),
    ("R031", "Vehicle", "V2-small", "runs/R031_ablation_vehicle_V2_small"),
    ("R032", "Uncertainty", "U0", "runs/R032_ablation_uncertainty_U0"),
    ("R033", "Uncertainty", "U1", "runs/R033_ablation_uncertainty_U1"),
]


def main() -> int:
    rows = []
    for run_id, family, variant, run_dir in RUNS:
        report_path = os.path.join(run_dir, "artifacts", "base_hybrid_report.json")
        summary_path = os.path.join(run_dir, "summary.json")
        if not os.path.exists(report_path):
            rows.append(_missing_row(run_id, family, variant))
            continue
        with open(report_path, "r", encoding="utf-8") as handle:
            report = json.load(handle)
        with open(summary_path, "r", encoding="utf-8") as handle:
            summary = json.load(handle)
        metrics = report["summary_metrics"]
        rows.append(
            {
                "run_id": run_id,
                "family": family,
                "variant": variant,
                "status": summary["status"],
                "passed": int(bool(summary["success_criteria_met"])),
                "validation_ratio": metrics.get(
                    "validation_base_vs_physics_ratio_h100_mean", 0.0
                ),
                "held_out_road_ratio": metrics.get(
                    "held_out_road_base_vs_physics_ratio_h100_mean", 0.0
                ),
                "held_out_vehicle_ratio": metrics.get(
                    "held_out_vehicle_base_vs_physics_ratio_h100_mean", 0.0
                ),
                "residual_ratio": metrics.get("base_residual_to_physics_ratio_mean", 0.0),
                "constraint_rate": metrics.get("base_constraint_violation_rate_mean", 0.0),
                "uncertainty_ood_lift": metrics.get("uncertainty_ood_lift_proxy", 0.0),
                "teacher_oracle": _teacher_oracle(report),
            }
        )
    os.makedirs("reports", exist_ok=True)
    _write_json("reports/B4_ablation_summary.json", {"rows": rows})
    _write_markdown("reports/B4_ablations.md", rows)
    print("wrote reports/B4_ablations.md")
    return 0


def _missing_row(run_id: str, family: str, variant: str) -> Dict[str, Any]:
    return {
        "run_id": run_id,
        "family": family,
        "variant": variant,
        "status": "missing",
        "passed": 0,
        "validation_ratio": 0.0,
        "held_out_road_ratio": 0.0,
        "held_out_vehicle_ratio": 0.0,
        "residual_ratio": 0.0,
        "constraint_rate": 0.0,
        "uncertainty_ood_lift": 0.0,
        "teacher_oracle": 0,
    }


def _teacher_oracle(report: Dict[str, Any]) -> int:
    variant = report.get("variant", {})
    return int(variant.get("mu") == "M2-oracle" or variant.get("fz") == "F2")


def _write_json(path: str, data: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)


def _write_markdown(path: str, rows: List[Dict[str, Any]]) -> None:
    lines = [
        "# B4 Component Ablation Scaffold Report",
        "",
        "This report aggregates M6 scaffold runs R015-R033. Ratios below are 100-step RMSE relative to physics-only; lower is better. These are scaffold-level comparisons, not final PyTorch module results.",
        "",
        "| Run | Family | Variant | Passed | Val / Phys | Held-out Road / Phys | Held-out Vehicle / Phys | Residual / Physics | Constraint Rate | Teacher Oracle |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {run_id} | {family} | {variant} | {passed} | {validation_ratio:.6f} | {held_out_road_ratio:.6f} | {held_out_vehicle_ratio:.6f} | {residual_ratio:.6f} | {constraint_rate:.6f} | {teacher_oracle} |".format(
                **row
            )
        )
    lines.extend(["", "## Family Winners", ""])
    for family in ["Tire", "Fz", "Steering", "Mu", "Encoder", "Vehicle", "Uncertainty"]:
        candidates = [
            row
            for row in rows
            if row["family"] == family and row["passed"] and not row["teacher_oracle"]
        ]
        if not candidates:
            continue
        best = min(candidates, key=lambda row: row["validation_ratio"])
        lines.append(
            "- %s: `%s` has the lowest validation ratio among deployable scaffold variants (`%.6f`)."
            % (family, best["variant"], best["validation_ratio"])
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `F2` and `M2-oracle` use teacher-only labels/features as simulator-only upper-bound or auxiliary scaffolds.",
            "- A passed ablation run means the comparison completed under the shared DS1 split and emitted metrics; it does not mean the variant is selected for the final model.",
            "- Final component selection still needs the full PyTorch implementation and multi-seed confirmation on training-scale DS1.",
            "",
        ]
    )
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


if __name__ == "__main__":
    raise SystemExit(main())
