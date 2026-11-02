"""Write a Harbor task tree from grok-scale rows."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from octopus.envs.swe.data.oracles import ORACLES
from octopus.envs.swe.taskset import load_rows, select_task_specs, verifier_shell, write_task_tree


def apply_oracle(spec: dict) -> tuple[dict, dict]:
    """Oracle tasks grade one behavior. Their instruction is that behavior."""
    extra = ORACLES.get(spec["instance_id"], {})
    if not extra:
        return spec, {}
    spec = dict(spec)
    if extra.get("instruction"):
        spec["instruction"] = extra["instruction"]
    return spec, extra


def emit(rows_dir: Path, images: set[str], excluded_ids: set[str], dest: Path) -> list[dict]:
    rows = load_rows(rows_dir)
    specs = select_task_specs(rows, images, excluded_ids)
    written = []
    for spec in specs:
        spec, extra = apply_oracle(spec)
        write_task_tree(
            dest,
            spec,
            **({"check_py": extra["check_py"], "fix_patch": extra["fix_patch"]} if extra else {}),
        )
        written.append(spec)
    (dest / "manifest.json").write_text(json.dumps(
        [{"instance_id": spec["instance_id"], "image": spec["image"]} for spec in written],
        indent=2,
    ) + "\n")
    return written


def emit_one(rows_dir: Path, instance_id: str, dest: Path) -> Path:
    for row in load_rows(rows_dir):
        task = row["solver_task"]
        if task["instance_id"] != instance_id:
            continue
        spec = {
            "instance_id": instance_id,
            "image": task["image"],
            "repo": task.get("repo") or "",
            "instruction": task["issue"],
            "verifier_shell": verifier_shell(instance_id),
        }
        spec, extra = apply_oracle(spec)
        return write_task_tree(
            dest, spec,
            **({"check_py": extra["check_py"], "fix_patch": extra["fix_patch"]} if extra else {}),
        )
    raise SystemExit(f"missing {instance_id}")


if __name__ == "__main__":
    rows_dir, image_file, excluded_file, dest = map(Path, sys.argv[1:5])
    images = set(json.loads(image_file.read_text()))
    excluded = set(json.loads(excluded_file.read_text()))
    specs = emit(rows_dir, images, excluded, dest)
    print(len(specs))
