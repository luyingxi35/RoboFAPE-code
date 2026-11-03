"""Portable manifests and strict trajectory validation."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import tempfile
import numpy as np

from .labeling import validate_profile


def read_json(path):
    def reject(value):
        raise ValueError(f"non-finite JSON number: {value}")
    return json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=reject)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        try:
            json.dump(value, handle, indent=2, allow_nan=False)
            handle.write("\n")
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
    os.replace(temporary, path)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def relative_path(value):
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise ValueError("artifact paths must be nonempty relative POSIX paths")
    path = PurePosixPath(value)
    if path.is_absolute() or PureWindowsPath(value).drive or ".." in path.parts or not path.parts:
        raise ValueError("artifact paths must be relative and contain no parent traversal")
    return path


def resource(root, relative):
    relative = relative_path(relative)
    root = Path(root).resolve()
    result = (root / relative).resolve()
    if not result.is_relative_to(root):
        raise ValueError("artifact path escapes manifest directory")
    if not result.is_file():
        raise FileNotFoundError(result)
    return result


def validate_trajectory(path):
    path = Path(path)
    data = read_json(path)
    values, _ = validate_profile(data)
    length = len(values)
    if "progress" in data:
        p = np.asarray(data["progress"], dtype=float)
        if p.shape != (length,) or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
            raise ValueError("progress must have one finite [0,1] value per frame")
    if data.get("artifact_kind") == "collected_trajectory":
        for field in ("actions", "trace", "frames"):
            if len(data.get(field, [])) != length:
                raise ValueError(f"{field} length differs from distance")
        actions = np.asarray(data["actions"], dtype=float)
        if actions.ndim != 2 or not np.isfinite(actions).all():
            raise ValueError("invalid actions")
        if len(data.get("control_modes",[])) != length:
            raise ValueError("one control mode is required per executed action")
        if len(set(data["frames"])) != length:
            raise ValueError("frame paths must be unique")
        if any(type(row.get("success")) is not bool for row in data["trace"]):
            raise ValueError("each trace row must record the native success predicate")
        if any(row["success"] for row in data["trace"]) != data["success"]:
            raise ValueError("recorded success disagrees with physical trace")
        for frame in data["frames"]:
            resource(path.parent, frame)
        resource(path.parent, data["states_file"])
        if data.get("video"):
            resource(path.parent, data["video"])
        import h5py
        with h5py.File(resource(path.parent, data["states_file"]), "r") as handle:
            if set(handle["states"]) != {str(i) for i in range(length+1)}:
                raise ValueError("state count must equal action count + 1")
        if type(data.get("success_once")) is not bool:
            raise ValueError("success_once is required")
        if data.get("source") == "policy" and not data["success"] and not data.get("policy_steps", 0):
            raise ValueError("failure requires at least one learned-policy step")
    for relative, expected in data.get("sha256", {}).items():
        if digest(resource(path.parent, relative)) != expected:
            raise ValueError(f"checksum mismatch: {relative}")
    return {"id": data["id"], "frames": length, "success": data["success"], "valid": True}
