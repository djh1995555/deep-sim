import argparse
import json
import os
from typing import Any, Dict

from teacher_simulator.config import load_teacher_config, load_yaml
from teacher_simulator.export import export_dataset
from teacher_simulator.scenario import make_ds0_scenarios
from teacher_simulator.simulator import TeacherSimulator


def generate_dataset(config_path: str, out_dir: str) -> Dict[str, Any]:
    raw = load_yaml(config_path)
    cfg = load_teacher_config(config_path)
    sim = TeacherSimulator(cfg)
    scenario_set = raw.get("scenario_set", "ds0")
    if scenario_set != "ds0":
        raise ValueError("v0 generator currently supports scenario_set=ds0")
    episodes = [sim.run_episode(scenario) for scenario in make_ds0_scenarios(cfg.seed)]
    manifest = export_dataset(
        episodes,
        out_dir,
        dataset_id=cfg.dataset_id,
        schema_version=cfg.schema_version,
        teacher_model_version=cfg.teacher_model_version,
    )
    with open(os.path.join(out_dir, "generation_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(
            {
                "config": config_path,
                "out_dir": out_dir,
                "episode_count": len(episodes),
                "scenario_set": scenario_set,
            },
            handle,
            indent=2,
            sort_keys=True,
        )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    try:
        manifest = generate_dataset(args.config, args.out)
    except ValueError as exc:
        print("configuration error: %s" % exc)
        return 2
    except FloatingPointError as exc:
        print("numerical integration failure: %s" % exc)
        return 3
    print(json.dumps({"status": "success", "episodes": len(manifest["episodes"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
