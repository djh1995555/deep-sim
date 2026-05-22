import argparse
import json
import os
from copy import deepcopy
from typing import Any, Dict, List

from simulator.vehicle_model.config import write_yaml


TRAINING_OUTPUT_ROOT = "output/training"
CONFIG_ROOT = "configs/experiments"
MATRIX_MANIFEST = os.path.join(CONFIG_ROOT, "matrix", "MANIFEST.json")
BASE_MODEL_CHECKPOINT = os.path.join(
    TRAINING_OUTPUT_ROOT,
    "R111_pytorch_base_model_small_training",
    "checkpoints",
    "base_model_small.pt",
)


BASE_MODEL_CONFIG = {
    "hidden_dim": 64,
    "history_len": 8,
    "encoder_type": "tcn",
    "tire_mode": "T1",
    "tire_projection": True,
    "fz_mode": "F1",
    "steering_mode": "S1",
    "mu_mode": "M1a",
    "vehicle_mode": "V1",
}


def _base_config(
    run_id: str,
    name: str,
    output_dir: str,
    mode: str,
    model_config: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "run": {
            "id": run_id,
            "name": name,
            "milestone": "P8",
            "experiment_block": "P8.generated",
            "seed": 23,
        },
        "data": {
            "dataset_id": "DS1_V1",
            "dataset_path": "data/ds1_v1",
            "schema_version": "v0",
            "split_id": "ds1_v1_scaffold",
            "train_filter": "train",
            "val_filter": "validation",
            "test_filter": "test",
        },
        "model": {
            "system_name": name,
            "student_config": "generated_pytorch_matrix",
            "baseline_config": None,
        },
        "training": {
            "optimizer": "adamw",
            "learning_rate": 0.0001,
            "batch_size": 16,
            "rollout_horizon": 1,
            "max_steps": 400,
            "early_stopping_metric": "torch_one_step_training_passed",
            "early_stopping_patience": 0,
        },
        "evaluation": {
            "horizons": ["1-step", "16-step"],
            "metrics": ["torch_one_step_training_passed", "torch_one_step_training_val_mse"],
            "report_splits": ["train", "validation"],
        },
        "logging": {
            "output_dir": output_dir,
            "save_checkpoints": True,
            "save_diagnostics": True,
        },
        "dataset_config": "configs/datasets/ds1_v1.yaml",
        "dataset_source": "existing",
        "artifact_dataset_subdir": "ds1",
        "primary_metric": "torch_one_step_training_passed",
        "torch_training": {
            "name": name,
            "mode": mode,
            "seed": 23,
            "split_role": "train",
            "val_split_role": "validation",
            "history_len": 8,
            "batch_size": 16,
            "max_episodes": 32,
            "max_samples": 1024,
            "max_val_samples": 512,
            "sample_stride": 1,
            "max_steps": 400,
            "eval_interval": 100,
            "learning_rate": 0.0001,
            "weight_decay": 0.0,
            "grad_clip": 1.0,
            "success_loss_ratio": 1.0,
            "loss_mode": "normalized_mse",
            "checkpoint_name": "model.pt",
            "device": "cuda",
            "require_cuda": True,
            "model_config": model_config,
        },
    }


def _ablation_specs() -> List[Dict[str, Any]]:
    factors = [
        ("encoder_E1", {"encoder_type": "gru"}),
        ("encoder_E2", {"encoder_type": "tcn"}),
        ("encoder_E3", {"encoder_type": "transformer"}),
        ("tire_T0", {"tire_mode": "T0"}),
        ("tire_T1", {"tire_mode": "T1", "tire_projection": True}),
        ("tire_T1_no_proj", {"tire_mode": "T1", "tire_projection": False}),
        ("tire_T2", {"tire_mode": "T2"}),
        ("fz_F0", {"fz_mode": "F0"}),
        ("fz_F1", {"fz_mode": "F1"}),
        ("steer_S0", {"steering_mode": "S0"}),
        ("steer_S1", {"steering_mode": "S1"}),
        ("mu_M0", {"mu_mode": "M0-fixed"}),
        ("mu_M1a", {"mu_mode": "M1a"}),
        ("vehicle_V0", {"vehicle_mode": "V0"}),
        ("vehicle_V1", {"vehicle_mode": "V1"}),
        ("vehicle_V1_large", {"vehicle_mode": "V1-large"}),
    ]
    specs = []
    for offset, (name, override) in enumerate(factors):
        run_id = "R%d" % (200 + offset)
        model_config = deepcopy(BASE_MODEL_CONFIG)
        model_config.update(override)
        specs.append(
            {
                "run_id": run_id,
                "name": "pytorch_ablation_%s" % name,
                "path": os.path.join(
                    CONFIG_ROOT,
                    "ablation",
                    "pytorch_ablation_%s.yaml" % name,
                ),
                "config": _base_config(
                    run_id,
                    "pytorch_ablation_%s" % name,
                    os.path.join(TRAINING_OUTPUT_ROOT, "%s_pytorch_ablation_%s" % (run_id, name)),
                    "one_step_train",
                    model_config,
                ),
            }
        )
    ensemble = _base_config(
        "R216",
        "pytorch_ablation_uncertainty_U1_ensemble",
        os.path.join(TRAINING_OUTPUT_ROOT, "R216_pytorch_ablation_uncertainty_U1_ensemble"),
        "ensemble_train",
        deepcopy(BASE_MODEL_CONFIG),
    )
    ensemble["primary_metric"] = "torch_ensemble_training_passed"
    ensemble["training"]["early_stopping_metric"] = "torch_ensemble_training_passed"
    ensemble["evaluation"]["metrics"] = [
        "torch_ensemble_training_passed",
        "torch_ensemble_mse",
        "torch_ensemble_predictive_variance",
    ]
    ensemble["torch_training"]["mode"] = "ensemble_train"
    ensemble["torch_training"]["ensemble_k"] = 3
    ensemble["torch_training"]["max_steps"] = 200
    specs.append(
        {
            "run_id": "R216",
            "name": "pytorch_ablation_uncertainty_U1_ensemble",
            "path": os.path.join(
                CONFIG_ROOT,
                "ablation",
                "pytorch_ablation_uncertainty_U1_ensemble.yaml",
            ),
            "config": ensemble,
        }
    )
    return specs


def _fine_tune_specs() -> List[Dict[str, Any]]:
    specs = []
    modes = ["FT0", "FT1", "FT2", "FT3", "FT4", "FT5", "FT6"]
    buckets = [
        ["FTD1"],
        ["FTD1", "FTD2"],
        ["FTD1", "FTD2", "FTD3"],
        ["FTD1", "FTD2", "FTD3", "FTD4"],
        ["FTD1", "FTD2", "FTD3", "FTD4", "FTD5"],
    ]
    run_num = 300
    for mode in modes:
        for bucket_index, bucket_values in enumerate(buckets):
            run_id = "R%d" % run_num
            cfg = _base_config(
                run_id,
                "pytorch_finetune_%s_B%d" % (mode, bucket_index),
                os.path.join(TRAINING_OUTPUT_ROOT, "%s_pytorch_finetune_%s_B%d" % (run_id, mode, bucket_index)),
                "one_step_train",
                deepcopy(BASE_MODEL_CONFIG),
            )
            cfg["run"]["milestone"] = "P9"
            cfg["run"]["experiment_block"] = "P9.generated"
            cfg["data"]["dataset_id"] = "DS1_PROXY_FT_V1"
            cfg["data"]["dataset_path"] = "data/ds1_proxy_ft_v1"
            cfg["data"]["split_id"] = "ds1_proxy_ft_v1_scaffold"
            cfg["data"]["train_filter"] = "fine-tune"
            cfg["data"]["val_filter"] = "test-window"
            cfg["data"]["test_filter"] = "test-window"
            cfg["dataset_config"] = "configs/datasets/ds1_proxy_ft_v1.yaml"
            cfg["artifact_dataset_subdir"] = "ds1_proxy_ft"
            cfg["torch_training"].update(
                {
                    "split_role": "fine-tune",
                    "val_split_role": "test-window",
                    "target_window_role": "target_train",
                    "val_target_window_role": "target_test",
                    "fine_tune_mode": mode,
                    "fine_tune_buckets": bucket_values,
                    "resume_from": BASE_MODEL_CHECKPOINT,
                    "max_steps": 0 if mode == "FT0" else 200,
                    "success_loss_ratio": 1.1,
                }
            )
            specs.append(
                {
                    "run_id": run_id,
                    "name": cfg["run"]["name"],
                    "path": os.path.join(
                        CONFIG_ROOT,
                        "finetune",
                        "%s.yaml" % cfg["run"]["name"],
                    ),
                    "config": cfg,
                }
            )
            run_num += 1
    return specs


def build_matrix() -> Dict[str, Any]:
    specs = _ablation_specs() + _fine_tune_specs()
    return {
        "config_count": len(specs),
        "ablation_count": len(_ablation_specs()),
        "fine_tune_count": len(_fine_tune_specs()),
        "configs": [{"run_id": item["run_id"], "name": item["name"], "path": item["path"]} for item in specs],
        "_specs": specs,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    matrix = build_matrix()
    specs = matrix.pop("_specs")
    if args.write:
        for item in specs:
            os.makedirs(os.path.dirname(item["path"]), exist_ok=True)
            write_yaml(item["path"], item["config"])
    os.makedirs(os.path.dirname(MATRIX_MANIFEST), exist_ok=True)
    with open(MATRIX_MANIFEST, "w", encoding="utf-8") as handle:
        json.dump(matrix, handle, indent=2, sort_keys=True)
    print("wrote %s" % MATRIX_MANIFEST)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
