import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List

from teacher_simulator.config import load_yaml, write_yaml
from teacher_simulator.generate import generate_dataset
from teacher_simulator.validators import TeacherEpisodeValidator, write_validation_report


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _run_shell(args: List[str]) -> str:
    try:
        proc = subprocess.run(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
            encoding="utf-8",
        )
        return proc.stdout
    except FileNotFoundError:
        return "%s not found\n" % args[0]


def _write_text(path: str, content: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(content)


def _write_json(path: str, data: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)


def _env_snapshot() -> str:
    lines = [
        "CONDA_DEFAULT_ENV=%s" % os.environ.get("CONDA_DEFAULT_ENV", ""),
        "CONDA_PREFIX=%s" % os.environ.get("CONDA_PREFIX", ""),
        "python_executable=%s" % sys.executable,
        "python=%s" % sys.version.replace("\n", " "),
        "platform=%s" % platform.platform(),
        "",
        "[pip freeze]",
        _run_shell([sys.executable, "-m", "pip", "freeze"]),
        "",
        "[nvidia-smi -L]",
        _run_shell(["nvidia-smi", "-L"]),
    ]
    return "\n".join(lines)


def _env_error() -> str:
    expected = os.environ.get("DEEP_SIM_CONDA_ENV", "deep-sim")
    current = os.environ.get("CONDA_DEFAULT_ENV", "")
    prefix = os.environ.get("CONDA_PREFIX", "")
    if current != expected:
        return "CONDA_DEFAULT_ENV must be %s, got %s" % (expected, current or "<empty>")
    if not prefix:
        return "CONDA_PREFIX is empty"
    if prefix not in sys.executable:
        return "python executable is outside CONDA_PREFIX"
    return ""


def _append_metric(path: str, run_id: str, metric_name: str, value: Any) -> None:
    row = {
        "step": 0,
        "split": "train",
        "horizon": "none",
        "metric_name": metric_name,
        "value": value,
        "state_channel": "none",
        "scenario_group": "DS0",
        "vehicle_group": "vehicle_A_debug",
        "seed": None,
        "timestamp": _now(),
        "run_id": run_id,
    }
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def run_config(config_path: str) -> int:
    cfg = load_yaml(config_path)
    run_cfg = cfg["run"]
    run_id = run_cfg["id"]
    out_dir = cfg["logging"]["output_dir"]
    start_time = _now()
    os.makedirs(out_dir, exist_ok=True)
    for sub in ["artifacts", "logs", "checkpoints"]:
        os.makedirs(os.path.join(out_dir, sub), exist_ok=True)
    shutil.copyfile(config_path, os.path.join(out_dir, "config.yaml"))
    write_yaml(os.path.join(out_dir, "resolved_config.yaml"), cfg)
    _write_text(os.path.join(out_dir, "command.txt"), " ".join(sys.argv) + "\n")
    _write_text(
        os.path.join(out_dir, "env.txt"),
        _env_snapshot(),
    )
    _write_text(os.path.join(out_dir, "git_status.txt"), _run_shell(["git", "status", "--short"]))

    dataset_dir = os.path.join(out_dir, "artifacts", "ds0")
    status = "success"
    notes = ""
    success = False
    env_error = _env_error()
    if env_error:
        summary = {
            "run_id": run_id,
            "milestone": run_cfg["milestone"],
            "experiment_block": run_cfg["experiment_block"],
            "config_name": run_cfg["name"],
            "status": "blocked",
            "start_time": start_time,
            "end_time": _now(),
            "dataset_id": cfg["data"]["dataset_id"],
            "train_split": cfg["data"]["split_id"],
            "val_split": cfg["data"].get("val_split", cfg["data"].get("val_filter")),
            "test_split": cfg["data"].get("test_split", cfg["data"].get("test_filter")),
            "seed": run_cfg["seed"],
            "primary_metric": cfg.get("primary_metric", "environment_ready"),
            "primary_metric_value": 0,
            "success_criteria_met": False,
            "notes": env_error,
        }
        _write_json(os.path.join(out_dir, "summary.json"), summary)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 1

    try:
        generate_dataset(cfg["teacher_config"], dataset_dir)
        validator = TeacherEpisodeValidator()
        report = validator.validate_dataset(dataset_dir)
        report_path = os.path.join(out_dir, "artifacts", "validation_report.json")
        write_validation_report(report, report_path)
        primary_metric, primary_value = _primary_metric(run_id, report.to_dict())
        metrics_path = os.path.join(out_dir, "metrics.jsonl")
        for name, value in report.metrics.items():
            _append_metric(metrics_path, run_id, name, value)
        _append_metric(metrics_path, run_id, primary_metric, primary_value)
        success = bool(report.passed and _primary_success(run_id, report.to_dict()))
        if not report.passed:
            status = "failed"
            notes = "; ".join(report.errors)
        elif not success:
            status = "failed"
            notes = "primary success gate failed"
        else:
            notes = "R000-R000d sanity subset gate passed for this run"
    except Exception as exc:
        status = "failed"
        primary_metric = cfg.get("primary_metric", "run_completed")
        primary_value = 0
        notes = repr(exc)

    summary = {
        "run_id": run_id,
        "milestone": run_cfg["milestone"],
        "experiment_block": run_cfg["experiment_block"],
        "config_name": run_cfg["name"],
        "status": status,
        "start_time": start_time,
        "end_time": _now(),
        "dataset_id": cfg["data"]["dataset_id"],
        "train_split": cfg["data"]["split_id"],
        "val_split": cfg["data"].get("val_split", cfg["data"].get("val_filter")),
        "test_split": cfg["data"].get("test_split", cfg["data"].get("test_filter")),
        "seed": run_cfg["seed"],
        "primary_metric": primary_metric,
        "primary_metric_value": primary_value,
        "success_criteria_met": bool(success),
        "notes": notes,
    }
    _write_json(os.path.join(out_dir, "summary.json"), summary)
    _write_b0_report()
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if success else 1


def _primary_metric(run_id: str, report: Dict[str, Any]) -> Any:
    metrics = report["metrics"]
    if run_id == "R000":
        return "episodes_generated", metrics.get("episode_count", 0)
    if run_id == "R000a":
        return "sign_checks_passed", metrics.get("sign_checks_passed", 0)
    if run_id == "R000b":
        score = (
            metrics.get("has_single_mu", 0)
            + metrics.get("has_split_mu", 0)
            + metrics.get("has_transition_mu", 0)
        )
        return "road_coverage_score", score
    if run_id == "R000c":
        return "sensor_actuator_checks_passed", int(report["passed"])
    if run_id == "R000d":
        return "leakage_checks_passed", metrics.get("leakage_checks_passed", 0)
    return "schema_checks_passed", metrics.get("schema_checks_passed", 0)


def _primary_success(run_id: str, report: Dict[str, Any]) -> bool:
    metrics = report["metrics"]
    if run_id == "R000":
        return metrics.get("episode_count", 0) >= 5
    if run_id == "R000a":
        return metrics.get("sign_checks_passed", 0) == 1
    if run_id == "R000b":
        return (
            metrics.get("has_single_mu", 0)
            + metrics.get("has_split_mu", 0)
            + metrics.get("has_transition_mu", 0)
        ) == 3
    if run_id == "R000c":
        return report["passed"]
    if run_id == "R000d":
        return metrics.get("leakage_checks_passed", 0) == 1
    return report["passed"]


def _write_b0_report() -> None:
    os.makedirs("reports", exist_ok=True)
    run_dirs = [
        "runs/R000_teacher_simulator_minimal",
        "runs/R000a_tire_load_validation",
        "runs/R000b_road_scenario_generation",
        "runs/R000c_sensor_actuator_realism",
        "runs/R000d_dataset_export_split",
    ]
    rows = []
    for run_dir in run_dirs:
        summary_path = os.path.join(run_dir, "summary.json")
        if not os.path.exists(summary_path):
            continue
        with open(summary_path, "r", encoding="utf-8") as handle:
            summary = json.load(handle)
        rows.append(summary)
    lines = [
        "# B0 Teacher Simulator Sanity Report",
        "",
        "| Run | Status | Primary Metric | Value | Success | Notes |",
        "|---|---|---:|---:|---|---|",
    ]
    for row in rows:
        notes = str(row.get("notes", "")).replace("|", "/")
        rendered = dict(row)
        rendered["notes"] = notes
        lines.append(
            "| {run_id} | {status} | {primary_metric} | {primary_metric_value} | {success_criteria_met} | {notes} |".format(
                **rendered
            )
        )
    _write_text("reports/B0_teacher.md", "\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    return run_config(args.config)


if __name__ == "__main__":
    raise SystemExit(main())
