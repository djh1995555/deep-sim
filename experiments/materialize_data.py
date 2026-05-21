import argparse
import json
import os
import shutil
from typing import Dict


CANONICAL_DATASETS: Dict[str, str] = {
    "ds1_v1": "runs/R000g_dataset_split_generation/artifacts/ds1",
    "ds1_proxy_ft_v1": "runs/R038_finetune_FT0/artifacts/ds1_proxy",
}


def materialize_dataset(name: str, mode: str, data_root: str) -> str:
    if name not in CANONICAL_DATASETS:
        raise ValueError("unknown dataset %s" % name)
    src = os.path.abspath(CANONICAL_DATASETS[name])
    dst = os.path.abspath(os.path.join(data_root, name))
    if not os.path.exists(src):
        raise FileNotFoundError(src)
    if os.path.lexists(dst):
        if os.path.islink(dst) and os.path.realpath(dst) == src:
            return dst
        raise FileExistsError("%s already exists; remove it before rematerializing" % dst)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if mode == "symlink":
        rel_src = os.path.relpath(src, os.path.dirname(dst))
        os.symlink(rel_src, dst)
    elif mode == "copy":
        shutil.copytree(src, dst)
    else:
        raise ValueError("unsupported mode %s" % mode)
    return dst


def validate_dataset(path: str) -> Dict[str, int]:
    manifest_path = os.path.join(path, "manifest.json")
    split_path = os.path.join(path, "split_manifest.json")
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(manifest_path)
    with open(manifest_path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    split_roles = set()
    if os.path.exists(split_path):
        with open(split_path, "r", encoding="utf-8") as handle:
            split = json.load(handle)
        split_roles = set(split.get("split_roles", {}))
    missing = []
    for item in manifest.get("episodes", []):
        for key in ["path", "sidecar"]:
            item_path = os.path.join(path, item[key])
            if not os.path.exists(item_path):
                missing.append(item_path)
    if missing:
        raise FileNotFoundError("missing dataset files: %s" % missing[:5])
    return {
        "episode_count": len(manifest.get("episodes", [])),
        "split_role_count": len(split_roles),
        "has_manifest": int(os.path.exists(manifest_path)),
        "has_split_manifest": int(os.path.exists(split_path)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        action="append",
        choices=sorted(CANONICAL_DATASETS),
        help="Dataset to materialize. Omit to materialize all canonical datasets.",
    )
    parser.add_argument("--mode", choices=["symlink", "copy"], default="symlink")
    parser.add_argument("--data-root", default="data")
    args = parser.parse_args()
    names = args.dataset or sorted(CANONICAL_DATASETS)
    report = {}
    for name in names:
        path = materialize_dataset(name, args.mode, args.data_root)
        report[name] = validate_dataset(path)
        report[name]["path"] = path
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
