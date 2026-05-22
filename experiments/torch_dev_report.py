import json
import os
from typing import Any, Dict, List


RUN_DIRS = {
    "R112": "output/training/R112_pytorch_fair_small_comparison",
    "R113": "output/training/R113_pytorch_model_variant_smoke",
    "R114": "output/training/R114_pytorch_fine_tune_adapter_smoke",
    "R115": "output/training/R115_pytorch_deep_ensemble_smoke",
}


def _read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: str, data: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)


def _write_text(path: str, content: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(content)


def _summary_rows() -> List[Dict[str, Any]]:
    rows = []
    for run_id, run_dir in RUN_DIRS.items():
        summary_path = os.path.join(run_dir, "summary.json")
        if os.path.exists(summary_path):
            rows.append(_read_json(summary_path))
    return rows


def build_report() -> Dict[str, Any]:
    report: Dict[str, Any] = {"runs": _summary_rows()}
    fair_path = os.path.join(
        RUN_DIRS["R112"],
        "artifacts",
        "fair_compare_report.json",
    )
    if os.path.exists(fair_path):
        fair = _read_json(fair_path)
        report["fair_compare"] = fair
        variants = fair.get("variants", [])
        if variants:
            report["fair_compare_table"] = [
                {
                    "name": row["name"],
                    "model_type": row["model_type"],
                    "val_mse": row["val_mse"],
                    "val_normalized_mse": row["val_normalized_mse"],
                    "rollout_rmse": row["rollout_rmse"],
                }
                for row in variants
            ]
    variant_path = os.path.join(
        RUN_DIRS["R113"],
        "artifacts",
        "model_variant_smoke.json",
    )
    if os.path.exists(variant_path):
        variant = _read_json(variant_path)
        report["model_variant_count"] = len(variant.get("variants", []))
        report["model_variants"] = variant.get("variants", [])
    fine_tune_path = os.path.join(
        RUN_DIRS["R114"],
        "artifacts",
        "fine_tune_smoke_report.json",
    )
    if os.path.exists(fine_tune_path):
        fine_tune = _read_json(fine_tune_path)
        report["fine_tune_rows"] = fine_tune.get("rows", [])
    ensemble_path = os.path.join(
        RUN_DIRS["R115"],
        "artifacts",
        "ensemble_report.json",
    )
    if os.path.exists(ensemble_path):
        ensemble = _read_json(ensemble_path)
        report["ensemble"] = ensemble
    return report


def render_markdown(report: Dict[str, Any]) -> str:
    lines = [
        "# PyTorch Development Report",
        "",
        "## Run Status",
        "",
        "| Run | Status | Primary Metric | Value |",
        "|---|---|---:|---:|",
    ]
    for row in report.get("runs", []):
        lines.append(
            "| {run_id} | {status} | {primary_metric} | {primary_metric_value} |".format(
                **row
            )
        )
    if report.get("fair_compare_table"):
        lines.extend(
            [
                "",
                "## R112 Fair Small Comparison",
                "",
                "| Variant | Type | Val MSE | Val Normalized MSE | Rollout RMSE |",
                "|---|---|---:|---:|---:|",
            ]
        )
        for row in report["fair_compare_table"]:
            lines.append(
                "| {name} | {model_type} | {val_mse:.6f} | {val_normalized_mse:.6f} | {rollout_rmse:.6f} |".format(
                    **row
                )
            )
        metrics = report.get("fair_compare", {}).get("metrics", {})
        lines.extend(
            [
                "",
                "Interpretation:",
                "",
                "- `hybrid_vs_best_black_box_mse_ratio = %.6f`."
                % metrics.get("torch_fair_compare_hybrid_vs_best_black_box_mse_ratio", float("nan")),
                "- `hybrid_vs_best_black_box_rollout_ratio = %.6f`."
                % metrics.get("torch_fair_compare_hybrid_vs_best_black_box_rollout_ratio", float("nan")),
                "- This is a development-scale run; it validates fair comparison plumbing and highlights that the current hybrid still needs model/loss tuning before any superiority claim.",
            ]
        )
    if report.get("model_variants"):
        lines.extend(
            [
                "",
                "## R113 Variant Coverage",
                "",
                "- Variant forward checks passed for `%d` model/component variants."
                % report.get("model_variant_count", 0),
            ]
        )
    if report.get("fine_tune_rows"):
        lines.extend(
            [
                "",
                "## R114 Fine-Tune Adapter Smoke",
                "",
                "| Mode | Buckets | Trainable Params | Val MSE |",
                "|---|---|---:|---:|",
            ]
        )
        for row in report["fine_tune_rows"]:
            lines.append(
                "| {mode} | {buckets} | {trainable_parameter_count} | {val_mse:.6f} |".format(
                    mode=row["mode"],
                    buckets="+".join(row["buckets"]),
                    trainable_parameter_count=row["trainable_parameter_count"],
                    val_mse=row["val_mse"],
                )
            )
    if report.get("ensemble"):
        metrics = report["ensemble"].get("metrics", {})
        lines.extend(
            [
                "",
                "## R115 Deep Ensemble Smoke",
                "",
                "- Members: `%d`." % metrics.get("torch_ensemble_member_count", 0),
                "- Ensemble MSE: `%.6f`." % metrics.get("torch_ensemble_mse", float("nan")),
                "- Predictive variance: `%.6f`."
                % metrics.get("torch_ensemble_predictive_variance", float("nan")),
            ]
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    report = build_report()
    _write_json("output/training/reports/PYTORCH_DEV_REPORT.json", report)
    _write_text("output/training/reports/PYTORCH_DEV_REPORT.md", render_markdown(report))
    print("wrote output/training/reports/PYTORCH_DEV_REPORT.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
