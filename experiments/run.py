import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from typing import Any, Dict, List

from experiments.baselines import baseline_metrics_for_runner, run_baseline_suite
from experiments.fine_tune import fine_tune_metrics_for_runner, run_fine_tune_suite
from experiments.hybrid import base_hybrid_metrics_for_runner, run_base_hybrid_suite
from experiments.sanity import run_sanity_check
from experiments.torch_training import run_torch_training_suite
from simulator.data_generator.generate import generate_dataset
from simulator.vehicle_model.config import load_yaml, write_yaml
from simulator.vehicle_model.validators import TeacherEpisodeValidator, write_validation_report


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
        "scenario_group": "stage_a",
        "vehicle_group": "mixed",
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

    dataset_subdir = cfg.get("artifact_dataset_subdir")
    if not dataset_subdir:
        dataset_subdir = "ds1" if cfg["data"]["dataset_id"].startswith("DS1") else "ds0"
    dataset_dir = os.path.join(out_dir, "artifacts", dataset_subdir)
    status = "success"
    notes = ""
    success = False
    blocked_reason = ""
    error_traceback_path = ""
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
        dataset_source = cfg.get("dataset_source", "generate")
        if dataset_source == "existing":
            dataset_dir = cfg["data"]["dataset_path"]
            if not os.path.exists(dataset_dir):
                raise FileNotFoundError("dataset_path does not exist: %s" % dataset_dir)
            _write_json(
                os.path.join(out_dir, "artifacts", "dataset_pointer.json"),
                {
                    "dataset_source": "existing",
                    "dataset_path": dataset_dir,
                    "artifact_dataset_subdir": dataset_subdir,
                },
            )
        else:
            generate_dataset(_dataset_config_path(cfg), dataset_dir)
        validator = TeacherEpisodeValidator()
        report = validator.validate_dataset(dataset_dir)
        report_dict = report.to_dict()
        _augment_report_metrics(dataset_dir, report_dict)
        sanity_cfg = cfg.get("sanity_check")
        if sanity_cfg:
            sanity_report = run_sanity_check(
                dataset_dir,
                sanity_cfg["name"],
                sanity_cfg,
            )
            _write_json(
                os.path.join(out_dir, "artifacts", "sanity_report.json"),
                sanity_report,
            )
            report_dict["metrics"].update(sanity_report.get("metrics", {}))
            report_dict["warnings"].extend(sanity_report.get("warnings", []))
            if not sanity_report.get("passed", False):
                report_dict["passed"] = False
                report_dict["errors"].extend(sanity_report.get("errors", []))
        baseline_cfg = cfg.get("baseline")
        if baseline_cfg:
            baseline_report = run_baseline_suite(dataset_dir, baseline_cfg, out_dir)
            report_dict["metrics"].update(baseline_metrics_for_runner(baseline_report))
            if not baseline_report.get("passed", False):
                report_dict["passed"] = False
                report_dict["errors"].append("baseline suite failed")
        hybrid_cfg = cfg.get("hybrid")
        if hybrid_cfg:
            base_hybrid_report = run_base_hybrid_suite(dataset_dir, hybrid_cfg, out_dir)
            report_dict["metrics"].update(
                base_hybrid_metrics_for_runner(base_hybrid_report)
            )
            if not base_hybrid_report.get("passed", False):
                report_dict["passed"] = False
                report_dict["errors"].append("base hybrid suite failed")
        fine_tune_cfg = cfg.get("fine_tune")
        if fine_tune_cfg:
            fine_tune_report = run_fine_tune_suite(dataset_dir, fine_tune_cfg, out_dir)
            report_dict["metrics"].update(
                fine_tune_metrics_for_runner(fine_tune_report)
            )
            if not fine_tune_report.get("passed", False):
                report_dict["passed"] = False
                report_dict["errors"].append("fine-tune suite failed")
        torch_training_cfg = cfg.get("torch_training")
        if torch_training_cfg:
            torch_training_report = run_torch_training_suite(
                dataset_dir,
                torch_training_cfg,
                out_dir,
            )
            report_dict["metrics"].update(torch_training_report.get("metrics", {}))
            if torch_training_report.get("blocked"):
                blocked_reason = torch_training_report.get("blocked_reason", "")
            if not torch_training_report.get("passed", False):
                report_dict["passed"] = False
                report_dict["errors"].append(
                    torch_training_report.get("blocked_reason")
                    or "torch training suite failed"
                )
        report_path = os.path.join(out_dir, "artifacts", "validation_report.json")
        _write_json(report_path, report_dict)
        primary_metric, primary_value = _primary_metric(run_id, report_dict)
        metrics_path = os.path.join(out_dir, "metrics.jsonl")
        for name, value in report_dict["metrics"].items():
            _append_metric(metrics_path, run_id, name, value)
        _append_metric(metrics_path, run_id, primary_metric, primary_value)
        success = bool(report_dict["passed"] and _primary_success(run_id, report_dict))
        if not report_dict["passed"]:
            status = "blocked" if blocked_reason else "failed"
            notes = "; ".join(report_dict["errors"])
        elif not success:
            status = "failed"
            notes = "primary success gate failed"
        else:
            notes = "experiment gate passed for this run"
    except Exception as exc:
        status = "failed"
        primary_metric = cfg.get("primary_metric", "run_completed")
        primary_value = 0
        notes = repr(exc)
        error_traceback_path = os.path.join(out_dir, "artifacts", "error_traceback.txt")
        _write_text(error_traceback_path, traceback.format_exc())

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
    if error_traceback_path:
        summary["error_traceback_path"] = error_traceback_path
    _write_json(os.path.join(out_dir, "summary.json"), summary)
    _write_stage_a_report()
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
    if run_id == "R000e":
        return "full_matrix_count", metrics.get("full_matrix_count", 0)
    if run_id == "R000f":
        return "vehicle_config_count", metrics.get("vehicle_config_count", 0)
    if run_id == "R000g":
        return "split_roles_covered", metrics.get("split_roles_covered", 0)
    if run_id == "R000h":
        return "dataset_qa_passed", metrics.get("dataset_qa_passed", 0)
    if run_id == "R001":
        return "role_schema_checks_passed", metrics.get("role_schema_checks_passed", 0)
    if run_id == "R002":
        return "physical_consistency_passed", metrics.get("physical_consistency_passed", 0)
    if run_id == "R003":
        return "dt_alignment_passed", metrics.get("dt_alignment_passed", 0)
    if run_id == "R004":
        return "derived_quantities_passed", metrics.get("derived_quantities_passed", 0)
    if run_id == "R004a":
        return "tiny_learnability_passed", metrics.get("tiny_learnability_passed", 0)
    if run_id == "R004b":
        return "physics_rollout_smoke_passed", metrics.get("physics_rollout_smoke_passed", 0)
    if run_id == "R004c":
        return "proxy_profiles_passed", metrics.get("proxy_profiles_passed", 0)
    if run_id == "R004d":
        return "proxy_target_windows_passed", metrics.get("proxy_target_windows_passed", 0)
    if run_id == "R004e":
        return "proxy_distribution_passed", metrics.get("proxy_distribution_passed", 0)
    if run_id == "R005":
        return "physics_only_baseline_passed", metrics.get("physics_only_baseline_passed", 0)
    if run_id == "R006":
        return "black_box_baseline_passed", metrics.get("black_box_baseline_passed", 0)
    if run_id == "R007":
        return "baseline_fairness_passed", metrics.get("baseline_fairness_passed", 0)
    if run_id == "R008":
        return "baseline_report_passed", metrics.get("baseline_report_passed", 0)
    if run_id == "R009":
        return "base_hybrid_training_passed", metrics.get("base_hybrid_training_passed", 0)
    if run_id == "R010":
        return "base_seen_config_eval_passed", metrics.get("base_seen_config_eval_passed", 0)
    if run_id == "R011":
        return "base_held_out_road_eval_passed", metrics.get("base_held_out_road_eval_passed", 0)
    if run_id == "R012":
        return "base_held_out_vehicle_eval_passed", metrics.get("base_held_out_vehicle_eval_passed", 0)
    if run_id == "R013":
        return "base_residual_constraint_audit_passed", metrics.get("base_residual_constraint_audit_passed", 0)
    if run_id == "R014":
        return "base_seed_replication_passed", metrics.get("base_seed_replication_passed", 0)
    if _is_m6_ablation_run(run_id):
        return "ablation_run_passed", metrics.get("ablation_run_passed", 0)
    if run_id == "R034":
        return "base_held_out_vehicle_eval_passed", metrics.get("base_held_out_vehicle_eval_passed", 0)
    if run_id in {"R035", "R036"}:
        return "cross_generalization_passed", metrics.get("cross_generalization_passed", 0)
    if run_id == "R037":
        return "final_single_model_freeze_passed", metrics.get("final_single_model_freeze_passed", 0)
    if run_id in {"R038", "R039", "R040", "R041", "R042", "R043", "R044"}:
        return "fine_tune_run_passed", metrics.get("fine_tune_run_passed", 0)
    if run_id == "R045":
        return "fine_tune_summary_passed", metrics.get("fine_tune_summary_passed", 0)
    if run_id == "R100":
        return "torch_data_loader_smoke_passed", metrics.get(
            "torch_data_loader_smoke_passed",
            0,
        )
    if run_id == "R101":
        return "torch_forward_loss_smoke_passed", metrics.get(
            "torch_forward_loss_smoke_passed",
            0,
        )
    if run_id == "R102":
        return "torch_tiny_overfit_passed", metrics.get("torch_tiny_overfit_passed", 0)
    if run_id == "R103":
        return "torch_rollout_smoke_passed", metrics.get("torch_rollout_smoke_passed", 0)
    if run_id == "R104":
        return "torch_checkpoint_smoke_passed", metrics.get(
            "torch_checkpoint_smoke_passed",
            0,
        )
    if run_id == "R105":
        return "torch_forward_loss_smoke_passed", metrics.get(
            "torch_forward_loss_smoke_passed",
            0,
        )
    if run_id == "R106":
        return "torch_tiny_overfit_passed", metrics.get("torch_tiny_overfit_passed", 0)
    if run_id in {"R107", "R111"}:
        return "torch_one_step_training_passed", metrics.get(
            "torch_one_step_training_passed",
            0,
        )
    if run_id == "R108":
        return "torch_rollout_eval_passed", metrics.get("torch_rollout_eval_passed", 0)
    if run_id == "R109":
        return "torch_resume_eval_passed", metrics.get("torch_resume_eval_passed", 0)
    if run_id == "R110":
        return "torch_black_box_training_passed", metrics.get(
            "torch_black_box_training_passed",
            0,
        )
    if run_id == "R112":
        return "torch_fair_compare_passed", metrics.get("torch_fair_compare_passed", 0)
    if run_id == "R113":
        return "torch_model_variant_smoke_passed", metrics.get(
            "torch_model_variant_smoke_passed",
            0,
        )
    if run_id == "R114":
        return "torch_fine_tune_smoke_passed", metrics.get(
            "torch_fine_tune_smoke_passed",
            0,
        )
    if run_id == "R115":
        return "torch_ensemble_training_passed", metrics.get(
            "torch_ensemble_training_passed",
            0,
        )
    for metric_name in [
        "torch_ensemble_training_passed",
        "torch_fine_tune_smoke_passed",
        "torch_fair_compare_passed",
        "torch_model_variant_smoke_passed",
        "torch_black_box_training_passed",
        "torch_one_step_training_passed",
        "torch_rollout_eval_passed",
    ]:
        if metric_name in metrics:
            return metric_name, metrics.get(metric_name, 0)
    return "schema_checks_passed", metrics.get("schema_checks_passed", 0)


def _dataset_config_path(cfg: Dict[str, Any]) -> str:
    path = cfg.get("dataset_config") or cfg.get("teacher_config")
    if not path:
        raise KeyError("run config must define dataset_config")
    return path


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
    if run_id == "R000e":
        return (
            metrics.get("full_matrix_count", 0) == 700
            and metrics.get("has_cg_single", 0) == 1
            and metrics.get("has_cg_split", 0) == 1
            and metrics.get("has_cg_transition", 0) == 1
        )
    if run_id == "R000f":
        return metrics.get("vehicle_config_count", 0) >= 3
    if run_id == "R000g":
        return (
            metrics.get("split_roles_covered", 0) >= 5
            and metrics.get("has_held_out_vehicle_config", 0) == 1
            and metrics.get("has_fine_tune_windows", 0) == 1
        )
    if run_id == "R000h":
        return metrics.get("dataset_qa_passed", 0) == 1
    if run_id == "R001":
        return metrics.get("role_schema_checks_passed", 0) == 1
    if run_id == "R002":
        return metrics.get("physical_consistency_passed", 0) == 1
    if run_id == "R003":
        return metrics.get("dt_alignment_passed", 0) == 1
    if run_id == "R004":
        return metrics.get("derived_quantities_passed", 0) == 1
    if run_id == "R004a":
        return metrics.get("tiny_learnability_passed", 0) == 1
    if run_id == "R004b":
        return metrics.get("physics_rollout_smoke_passed", 0) == 1
    if run_id == "R004c":
        return (
            metrics.get("proxy_profiles_passed", 0) == 1
            and metrics.get("proxy_profile_count", 0) >= 3
        )
    if run_id == "R004d":
        return (
            metrics.get("proxy_target_windows_passed", 0) == 1
            and metrics.get("proxy_target_window_count", 0) >= 20
            and metrics.get("proxy_target_window_overlap_count", 1) == 0
        )
    if run_id == "R004e":
        return (
            metrics.get("proxy_distribution_passed", 0) == 1
            and metrics.get("proxy_distribution_shift_score", 0.0) >= 0.015
        )
    if run_id == "R005":
        return metrics.get("physics_only_baseline_passed", 0) == 1
    if run_id == "R006":
        return (
            metrics.get("black_box_baseline_passed", 0) == 1
            and metrics.get("black_box_variant_count", 0) >= 4
        )
    if run_id == "R007":
        return metrics.get("baseline_fairness_passed", 0) == 1
    if run_id == "R008":
        return metrics.get("baseline_report_passed", 0) == 1
    if run_id == "R009":
        return metrics.get("base_hybrid_training_passed", 0) == 1
    if run_id == "R010":
        return metrics.get("base_seen_config_eval_passed", 0) == 1
    if run_id == "R011":
        return metrics.get("base_held_out_road_eval_passed", 0) == 1
    if run_id == "R012":
        return metrics.get("base_held_out_vehicle_eval_passed", 0) == 1
    if run_id == "R013":
        return metrics.get("base_residual_constraint_audit_passed", 0) == 1
    if run_id == "R014":
        return metrics.get("base_seed_replication_passed", 0) == 1
    if _is_m6_ablation_run(run_id):
        return metrics.get("ablation_run_passed", 0) == 1
    if run_id == "R034":
        return metrics.get("base_held_out_vehicle_eval_passed", 0) == 1
    if run_id in {"R035", "R036"}:
        return metrics.get("cross_generalization_passed", 0) == 1
    if run_id == "R037":
        return metrics.get("final_single_model_freeze_passed", 0) == 1
    if run_id in {"R038", "R039", "R040", "R041", "R042", "R043", "R044"}:
        return metrics.get("fine_tune_run_passed", 0) == 1
    if run_id == "R045":
        return metrics.get("fine_tune_summary_passed", 0) == 1
    if run_id == "R100":
        return metrics.get("torch_data_loader_smoke_passed", 0) == 1
    if run_id == "R101":
        return metrics.get("torch_forward_loss_smoke_passed", 0) == 1
    if run_id == "R102":
        return metrics.get("torch_tiny_overfit_passed", 0) == 1
    if run_id == "R103":
        return metrics.get("torch_rollout_smoke_passed", 0) == 1
    if run_id == "R104":
        return metrics.get("torch_checkpoint_smoke_passed", 0) == 1
    if run_id == "R105":
        return metrics.get("torch_forward_loss_smoke_passed", 0) == 1
    if run_id == "R106":
        return metrics.get("torch_tiny_overfit_passed", 0) == 1
    if run_id in {"R107", "R111"}:
        return metrics.get("torch_one_step_training_passed", 0) == 1
    if run_id == "R108":
        return metrics.get("torch_rollout_eval_passed", 0) == 1
    if run_id == "R109":
        return metrics.get("torch_resume_eval_passed", 0) == 1
    if run_id == "R110":
        return metrics.get("torch_black_box_training_passed", 0) == 1
    if run_id == "R112":
        return metrics.get("torch_fair_compare_passed", 0) == 1
    if run_id == "R113":
        return metrics.get("torch_model_variant_smoke_passed", 0) == 1
    if run_id == "R114":
        return metrics.get("torch_fine_tune_smoke_passed", 0) == 1
    if run_id == "R115":
        return metrics.get("torch_ensemble_training_passed", 0) == 1
    for metric_name in [
        "torch_ensemble_training_passed",
        "torch_fine_tune_smoke_passed",
        "torch_fair_compare_passed",
        "torch_model_variant_smoke_passed",
        "torch_black_box_training_passed",
        "torch_one_step_training_passed",
        "torch_rollout_eval_passed",
    ]:
        if metric_name in metrics:
            return metrics.get(metric_name, 0) == 1
    return report["passed"]


def _is_m6_ablation_run(run_id: str) -> bool:
    return run_id in {
        "R015",
        "R016",
        "R017",
        "R018",
        "R019",
        "R020",
        "R021",
        "R022",
        "R023",
        "R024",
        "R025",
        "R026",
        "R027",
        "R027a",
        "R027b",
        "R027c",
        "R028",
        "R029",
        "R030",
        "R031",
        "R032",
        "R033",
    }


def _augment_report_metrics(dataset_dir: str, report: Dict[str, Any]) -> None:
    coverage_path = os.path.join(dataset_dir, "scenario_coverage_report.json")
    if os.path.exists(coverage_path):
        with open(coverage_path, "r", encoding="utf-8") as handle:
            coverage = json.load(handle)
        for key in [
            "full_matrix_count",
            "sampled_episode_count",
            "sampled_road_factor_count",
        ]:
            report["metrics"][key] = coverage.get(key, 0)
    qa_path = os.path.join(dataset_dir, "dataset_qa_report.json")
    if os.path.exists(qa_path):
        with open(qa_path, "r", encoding="utf-8") as handle:
            qa = json.load(handle)
        report["metrics"]["dataset_qa_passed"] = int(bool(qa.get("passed")))
        report["metrics"]["qa_vehicle_config_count"] = qa.get("vehicle_config_count", 0)
        report["metrics"]["qa_target_window_count"] = qa.get("target_window_count", 0)
    profile_path = os.path.join(dataset_dir, "perturbation_profiles.json")
    if os.path.exists(profile_path):
        with open(profile_path, "r", encoding="utf-8") as handle:
            profiles = json.load(handle)
        mins = [profile.get("min_abs_magnitude", 0.0) for profile in profiles]
        maxs = [profile.get("max_abs_magnitude", 0.0) for profile in profiles]
        report["metrics"]["proxy_profile_count"] = len(profiles)
        report["metrics"]["proxy_profile_min_abs_magnitude"] = min(mins) if mins else 0.0
        report["metrics"]["proxy_profile_max_abs_magnitude"] = max(maxs) if maxs else 0.0
        report["metrics"]["proxy_profiles_passed"] = int(
            bool(profiles)
            and min(mins) >= 0.05 - 1e-9
            and max(maxs) <= 0.15 + 1e-9
        )
    target_path = os.path.join(dataset_dir, "proxy_target_windows.json")
    if os.path.exists(target_path):
        with open(target_path, "r", encoding="utf-8") as handle:
            target_report = json.load(handle)
        report["metrics"]["proxy_target_windows_passed"] = int(
            bool(target_report.get("passed"))
        )
        report["metrics"]["proxy_target_window_count"] = target_report.get(
            "target_window_count", 0
        )
        report["metrics"]["proxy_target_window_overlap_count"] = target_report.get(
            "target_window_overlap_count", 0
        )
        report["metrics"]["proxy_held_out_vehicle_config_count"] = target_report.get(
            "held_out_vehicle_config_count", 0
        )
    distribution_path = os.path.join(dataset_dir, "proxy_distribution_report.json")
    if os.path.exists(distribution_path):
        with open(distribution_path, "r", encoding="utf-8") as handle:
            distribution = json.load(handle)
        report["metrics"]["proxy_distribution_passed"] = int(
            bool(distribution.get("passed"))
        )
        report["metrics"]["proxy_distribution_shift_score"] = distribution.get(
            "proxy_distribution_shift_score", 0.0
        )
        report["metrics"]["proxy_distribution_paired_episode_count"] = distribution.get(
            "paired_episode_count", 0
        )


def _write_stage_a_report() -> None:
    os.makedirs("output/training/reports", exist_ok=True)
    run_dirs = [
        "output/training/R000_dataset_generation_minimal",
        "output/training/R000a_tire_load_validation",
        "output/training/R000b_road_scenario_generation",
        "output/training/R000c_sensor_actuator_realism",
        "output/training/R000d_dataset_export_split",
        "output/training/R000e_scenario_matrix_v1",
        "output/training/R000f_vehicle_parameter_randomization",
        "output/training/R000g_dataset_split_generation",
        "output/training/R000h_dataset_qa",
        "output/training/R001_schema_field_role_check",
        "output/training/R002_teacher_physical_consistency",
        "output/training/R003_time_dt_alignment",
        "output/training/R004_derived_physical_quantities",
        "output/training/R004a_tiny_learnability",
        "output/training/R004b_physics_rollout_smoke",
        "output/training/R004c_proxy_perturbation_profiles",
        "output/training/R004d_proxy_target_windows",
        "output/training/R004e_proxy_distribution_sanity",
        "output/training/R005_physics_only_baseline",
        "output/training/R006_black_box_baseline",
        "output/training/R007_baseline_fairness_audit",
        "output/training/R008_baseline_rollout_report",
        "output/training/R009_base_hybrid_training",
        "output/training/R010_base_seen_config_eval",
        "output/training/R011_base_held_out_road_eval",
        "output/training/R012_base_held_out_vehicle_eval",
        "output/training/R013_base_residual_constraint_audit",
        "output/training/R014_base_seed_replication",
        "output/training/R015_ablation_tire_T0",
        "output/training/R016_ablation_tire_T1",
        "output/training/R017_ablation_tire_T1_no_proj",
        "output/training/R018_ablation_tire_T2",
        "output/training/R019_ablation_fz_F0",
        "output/training/R020_ablation_fz_F1",
        "output/training/R021_ablation_fz_F2",
        "output/training/R022_ablation_steering_S0",
        "output/training/R023_ablation_steering_S1",
        "output/training/R024_ablation_mu_M0_fixed",
        "output/training/R025_ablation_mu_M1a",
        "output/training/R026_ablation_mu_M1b",
        "output/training/R027_ablation_mu_M2_oracle",
        "output/training/R027a_ablation_encoder_E1",
        "output/training/R027b_ablation_encoder_E2",
        "output/training/R027c_ablation_encoder_E3",
        "output/training/R028_ablation_vehicle_V0",
        "output/training/R029_ablation_vehicle_V1",
        "output/training/R030_ablation_vehicle_V1_large",
        "output/training/R031_ablation_vehicle_V2_small",
        "output/training/R032_ablation_uncertainty_U0",
        "output/training/R033_ablation_uncertainty_U1",
        "output/training/R034_cross_generalization_base",
        "output/training/R035_cross_generalization_selected_single",
        "output/training/R036_cross_generalization_selected_ensemble",
        "output/training/R037_final_single_model_freeze",
        "output/training/R038_finetune_FT0",
        "output/training/R039_finetune_FT1_vehicle_param_adapter",
        "output/training/R040_finetune_FT2_mu_head",
        "output/training/R041_finetune_FT3_fz_residual",
        "output/training/R042_finetune_FT4_tire_residual",
        "output/training/R043_finetune_FT5_steering_residual",
        "output/training/R044_finetune_FT6_full_model",
        "output/training/R045_finetune_summary",
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
        "# Experiment Bridge Data/Physics/Baseline/Hybrid Report",
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
    _write_text("output/training/reports/B0_data_generation.md", "\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    return run_config(args.config)


if __name__ == "__main__":
    raise SystemExit(main())
