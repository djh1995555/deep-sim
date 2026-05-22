import argparse
import json
import os
from typing import Any, Dict, List

from simulator.vehicle_model.config import load_yaml


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


def _load_manifest(path: str) -> List[Dict[str, Any]]:
    manifest = _read_json(path)
    return manifest.get("configs", [])


def _safe_metrics(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    try:
        return _read_json(path).get("metrics", {})
    except json.JSONDecodeError:
        return {}


def _safe_summary(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    try:
        return _read_json(path)
    except json.JSONDecodeError:
        return {}


def _row_for_config(item: Dict[str, Any]) -> Dict[str, Any]:
    cfg = load_yaml(item["path"])
    out_dir = cfg["logging"]["output_dir"]
    summary = _safe_summary(os.path.join(out_dir, "summary.json"))
    metrics = _safe_metrics(os.path.join(out_dir, "artifacts", "validation_report.json"))
    rollout = _safe_summary(os.path.join(out_dir, "artifacts", "post_rollout_eval", "summary.json"))
    torch_cfg = cfg.get("torch_training", {})
    return {
        "run_id": item["run_id"],
        "name": item["name"],
        "config_path": item["path"],
        "output_dir": out_dir,
        "status": summary.get("status", "pending"),
        "success": bool(summary.get("success_criteria_met", False)),
        "primary_metric": summary.get("primary_metric", ""),
        "primary_metric_value": summary.get("primary_metric_value", None),
        "val_mse": metrics.get("torch_one_step_training_val_mse"),
        "val_normalized_mse": metrics.get("torch_one_step_training_val_normalized_mse"),
        "best_val_loss": metrics.get("torch_one_step_training_best_val_loss"),
        "completed_steps": metrics.get("torch_one_step_training_completed_steps"),
        "rollout_passed": rollout.get("passed"),
        "rollout_rmse": rollout.get("metrics", {}).get("torch_rollout_eval_rmse"),
        "rollout_constraint_violation_rate": rollout.get("metrics", {}).get(
            "torch_rollout_eval_constraint_violation_rate"
        ),
        "fine_tune_mode": torch_cfg.get("fine_tune_mode"),
        "fine_tune_buckets": torch_cfg.get("fine_tune_buckets", []),
        "model_config": torch_cfg.get("model_config", {}),
    }


def build_matrix_report(manifest_path: str) -> Dict[str, Any]:
    rows = [_row_for_config(item) for item in _load_manifest(manifest_path)]
    ablation_rows = [row for row in rows if 200 <= int(row["run_id"][1:]) <= 216]
    fine_tune_rows = [row for row in rows if 300 <= int(row["run_id"][1:]) <= 334]
    completed_ablations = [
        row for row in ablation_rows if row.get("success") and isinstance(row.get("val_normalized_mse"), (int, float))
    ]
    completed_ablations.sort(key=lambda row: row["val_normalized_mse"])
    completed_ft = [
        row for row in fine_tune_rows if row.get("success") and isinstance(row.get("val_mse"), (int, float))
    ]
    completed_ft.sort(key=lambda row: (row.get("fine_tune_mode") or "", len(row.get("fine_tune_buckets") or [])))
    status_counts: Dict[str, int] = {}
    for row in rows:
        status_counts[row["status"]] = status_counts.get(row["status"], 0) + 1
    return {
        "manifest_path": manifest_path,
        "status_counts": status_counts,
        "rows": rows,
        "ablation_rows": ablation_rows,
        "fine_tune_rows": fine_tune_rows,
        "ablation_ranking": completed_ablations,
        "fine_tune_completed": completed_ft,
    }


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return "%.6f" % value
    if value is None:
        return ""
    return str(value)


def render_markdown(report: Dict[str, Any]) -> str:
    lines = [
        "# PyTorch Matrix Report",
        "",
        "## Status",
        "",
        "| Status | Count |",
        "|---|---:|",
    ]
    for status, count in sorted(report["status_counts"].items()):
        lines.append("| %s | %d |" % (status, count))
    lines.extend(
        [
            "",
            "## Ablation Ranking",
            "",
            "| Run | Name | Success | Val Normalized MSE | Val MSE | Rollout RMSE |",
            "|---|---|---|---:|---:|---:|",
        ]
    )
    ranking = report.get("ablation_ranking") or report.get("ablation_rows", [])
    for row in ranking:
        lines.append(
            "| {run_id} | {name} | {success} | {norm} | {mse} | {rollout} |".format(
                run_id=row["run_id"],
                name=row["name"],
                success=row["success"],
                norm=_fmt(row.get("val_normalized_mse")),
                mse=_fmt(row.get("val_mse")),
                rollout=_fmt(row.get("rollout_rmse")),
            )
        )
    lines.extend(
        [
            "",
            "## Fine-Tune Data Efficiency",
            "",
            "| Run | Mode | Buckets | Success | Val MSE | Rollout RMSE |",
            "|---|---|---|---|---:|---:|",
        ]
    )
    fine_tune_rows = report.get("fine_tune_completed") or report.get("fine_tune_rows", [])
    for row in fine_tune_rows:
        lines.append(
            "| {run_id} | {mode} | {buckets} | {success} | {mse} | {rollout} |".format(
                run_id=row["run_id"],
                mode=row.get("fine_tune_mode") or "",
                buckets="+".join(row.get("fine_tune_buckets") or []),
                success=row["success"],
                mse=_fmt(row.get("val_mse")),
                rollout=_fmt(row.get("rollout_rmse")),
            )
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="configs/torch_matrix/MANIFEST.json")
    parser.add_argument("--json-out", default="reports/PYTORCH_MATRIX_REPORT.json")
    parser.add_argument("--md-out", default="reports/PYTORCH_MATRIX_REPORT.md")
    args = parser.parse_args()
    report = build_matrix_report(args.manifest)
    _write_json(args.json_out, report)
    _write_text(args.md_out, render_markdown(report))
    print("wrote %s" % args.md_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
