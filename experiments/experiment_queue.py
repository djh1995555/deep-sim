import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List

from experiments.torch_training import run_torch_training_suite
from simulator.vehicle_model.config import load_yaml


TRAINING_OUTPUT_ROOT = "output/training"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: str, data: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)


def _read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _summary_success(config_path: str) -> bool:
    cfg = load_yaml(config_path)
    summary_path = os.path.join(cfg["logging"]["output_dir"], "summary.json")
    if not os.path.exists(summary_path):
        return False
    try:
        summary = _read_json(summary_path)
    except json.JSONDecodeError:
        return False
    return bool(summary.get("success_criteria_met")) and summary.get("status") == "success"


def _rollout_success(config_path: str) -> bool:
    cfg = load_yaml(config_path)
    summary_path = os.path.join(
        cfg["logging"]["output_dir"],
        "artifacts",
        "post_rollout_eval",
        "summary.json",
    )
    if not os.path.exists(summary_path):
        return False
    try:
        summary = _read_json(summary_path)
    except json.JSONDecodeError:
        return False
    return bool(summary.get("passed"))


def _load_jobs(configs: List[str], manifest_path: str, run_ids: List[str]) -> List[Dict[str, Any]]:
    jobs: List[Dict[str, Any]] = []
    seen = set()
    if manifest_path:
        manifest = _read_json(manifest_path)
        allowed = set(run_ids)
        for item in manifest.get("configs", []):
            if allowed and item["run_id"] not in allowed:
                continue
            path = item["path"]
            if path not in seen:
                jobs.append({"run_id": item["run_id"], "config_path": path, "name": item.get("name", item["run_id"])})
                seen.add(path)
    for path in configs:
        cfg = load_yaml(path)
        run_id = cfg["run"]["id"]
        if run_ids and run_id not in set(run_ids):
            continue
        if path not in seen:
            jobs.append({"run_id": run_id, "config_path": path, "name": cfg["run"]["name"]})
            seen.add(path)
    jobs.sort(key=lambda item: item["run_id"])
    return jobs


def _run_config(config_path: str, log_path: str) -> int:
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    cmd = [sys.executable, "-m", "experiments.run", "--config", config_path]
    with open(log_path, "w", encoding="utf-8") as log:
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=False)
    return int(proc.returncode)


def _checkpoint_path_for_config(cfg: Dict[str, Any]) -> str:
    torch_cfg = cfg.get("torch_training", {})
    checkpoint_name = torch_cfg.get("checkpoint_name", "model.pt")
    return os.path.join(cfg["logging"]["output_dir"], "checkpoints", checkpoint_name)


def _run_post_rollout(config_path: str, rollout_steps: int, max_episodes: int) -> Dict[str, Any]:
    cfg = load_yaml(config_path)
    torch_cfg = cfg.get("torch_training", {})
    if torch_cfg.get("mode") != "one_step_train":
        return {"skipped": True, "reason": "rollout is only automatic for one_step_train configs"}
    dataset_dir = cfg["data"]["dataset_path"]
    checkpoint_path = _checkpoint_path_for_config(cfg)
    out_dir = os.path.join(cfg["logging"]["output_dir"], "artifacts", "post_rollout_eval")
    rollout_cfg = {
        **torch_cfg,
        "name": "%s_post_rollout_eval" % cfg["run"]["id"],
        "mode": "rollout_eval",
        "checkpoint_path": checkpoint_path,
        "split_role": torch_cfg.get("val_split_role", "validation"),
        "target_window_role": torch_cfg.get("val_target_window_role"),
        "fine_tune_buckets": torch_cfg.get("val_fine_tune_buckets"),
        "rollout_steps": rollout_steps,
        "max_episodes": max_episodes,
        "max_rollout_rmse": float(torch_cfg.get("max_rollout_rmse", 1.0e6)),
        "max_constraint_violation_rate": float(torch_cfg.get("max_constraint_violation_rate", 0.0)),
    }
    report = run_torch_training_suite(dataset_dir, rollout_cfg, out_dir)
    summary = {
        "run_id": cfg["run"]["id"],
        "config_path": config_path,
        "checkpoint_path": checkpoint_path,
        "passed": bool(report.get("passed")),
        "blocked": bool(report.get("blocked")),
        "metrics": report.get("metrics", {}),
        "end_time": _now(),
    }
    _write_json(os.path.join(out_dir, "summary.json"), summary)
    return summary


def run_queue(args: argparse.Namespace) -> Dict[str, Any]:
    jobs = _load_jobs(args.configs, args.manifest, args.run_ids)
    if args.limit > 0:
        jobs = jobs[: args.limit]
    state_path = args.state_path
    state: Dict[str, Any] = {
        "created_at": _now(),
        "updated_at": _now(),
        "dry_run": bool(args.dry_run),
        "jobs": {},
    }
    if os.path.exists(state_path) and not args.reset_state:
        state = _read_json(state_path)
        state.setdefault("jobs", {})
    for job in jobs:
        run_id = job["run_id"]
        config_path = job["config_path"]
        job_state = state["jobs"].setdefault(run_id, {"config_path": config_path, "attempts": 0})
        job_state.update({"config_path": config_path, "name": job["name"]})
        if args.skip_success and _summary_success(config_path):
            if args.rollout_eval and not _rollout_success(config_path):
                job_state["status"] = "success_needs_rollout"
                if args.dry_run:
                    job_state["updated_at"] = _now()
                    continue
                rollout = _run_post_rollout(config_path, args.rollout_steps, args.rollout_max_episodes)
                job_state["rollout"] = rollout
                job_state["status"] = "success" if rollout.get("skipped") or rollout.get("passed") else "failed_rollout"
                job_state["updated_at"] = _now()
                continue
            else:
                job_state["status"] = "skipped_success"
                job_state["updated_at"] = _now()
                continue
        if args.dry_run:
            job_state["status"] = "dry_run"
            job_state["updated_at"] = _now()
            continue
        success = False
        for attempt in range(int(job_state.get("attempts", 0)), args.max_retries + 1):
            job_state["attempts"] = attempt + 1
            job_state["status"] = "running"
            job_state["started_at"] = _now()
            log_path = os.path.join(args.log_dir, "%s_attempt_%d.log" % (run_id, attempt + 1))
            job_state["log_path"] = log_path
            state["updated_at"] = _now()
            _write_json(state_path, state)
            rc = _run_config(config_path, log_path)
            job_state["returncode"] = rc
            success = rc == 0 and _summary_success(config_path)
            job_state["status"] = "success" if success else "failed"
            job_state["updated_at"] = _now()
            _write_json(state_path, state)
            if success:
                break
        if success and args.rollout_eval:
            rollout = _run_post_rollout(config_path, args.rollout_steps, args.rollout_max_episodes)
            job_state["rollout"] = rollout
            if not rollout.get("skipped") and not rollout.get("passed"):
                job_state["status"] = "failed_rollout"
        state["updated_at"] = _now()
        _write_json(state_path, state)
        if not success and args.stop_on_failure:
            break
    state["updated_at"] = _now()
    _write_json(state_path, state)
    return state


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="")
    parser.add_argument("--configs", nargs="*", default=[])
    parser.add_argument("--run-ids", nargs="*", default=[])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument("--skip-success", action="store_true")
    parser.add_argument("--stop-on-failure", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--reset-state", action="store_true")
    parser.add_argument("--rollout-eval", action="store_true")
    parser.add_argument("--rollout-steps", type=int, default=16)
    parser.add_argument("--rollout-max-episodes", type=int, default=4)
    parser.add_argument("--state-path", default=os.path.join(TRAINING_OUTPUT_ROOT, "queue_state.json"))
    parser.add_argument("--log-dir", default=os.path.join(TRAINING_OUTPUT_ROOT, "_queue_logs"))
    args = parser.parse_args()
    state = run_queue(args)
    counts: Dict[str, int] = {}
    for job in state.get("jobs", {}).values():
        counts[job.get("status", "unknown")] = counts.get(job.get("status", "unknown"), 0) + 1
    print(json.dumps({"state_path": args.state_path, "counts": counts}, indent=2, sort_keys=True))
    failed = sum(count for status, count in counts.items() if status.startswith("failed"))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
