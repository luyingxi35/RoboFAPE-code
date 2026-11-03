"""Packaged, fixed task configuration (no online model calls)."""
from __future__ import annotations
from importlib.resources import files
import hashlib
import json
from pathlib import Path
from .expressions import SafeExpression


def task_ids():
    return sorted(p.name[:-5] for p in files("robofape").joinpath("configs").iterdir() if p.name.endswith(".json"))


def load_config(task, path=None):
    if path is None and task not in task_ids():
        raise ValueError(f"unknown task {task!r}; run robofape tasks")
    raw = (Path(path) if path else files("robofape").joinpath("configs", task + ".json")).read_text()
    config = json.loads(raw)
    if config["task_id"] != task:
        raise ValueError("configuration task_id mismatch")
    names = [o["name"] for o in config["observations"]]
    if len(set(names)) != len(names):
        raise ValueError("duplicate observation names")
    stages = config["stage_order"]
    if not stages or len(set(stages)) != len(stages):
        raise ValueError("stage_order must contain unique names")
    if config["profile_adapter"] == "schema":
        for stage in config["stages"]:
            SafeExpression(stage["metric"], set(names))
            SafeExpression(stage["end_condition"], set(names), allow_condition=True)
    config["sha256"] = hashlib.sha256(raw.encode()).hexdigest()
    return config
