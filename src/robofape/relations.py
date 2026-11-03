# Copyright (c) 2025 MINT-SJTU
# SPDX-License-Identifier: MIT
# Modified for package-local imports and standalone collection.
"""Whitelisted relations available to VLM-selected progress metrics."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np


RELATION_REGISTRY: dict[str, dict[str, Any]] = {
    "distance": {
        "input_types": ["pos", "pos"],
        "output_type": "float",
        "default_direction": "decrease",
        "description": "Euclidean distance between two positions.",
    },
    "squared_distance": {
        "input_types": ["pos", "pos"],
        "output_type": "float",
        "default_direction": "decrease",
        "description": "Squared Euclidean distance between two positions.",
    },
    "pose_error": {
        "input_types": ["pose", "pose"],
        "output_type": "float",
        "default_direction": "decrease",
        "description": "Position error plus quaternion angular error. Optional third scalar weights rotation.",
    },
    "relative_pose_norm": {
        "input_types": ["pos", "quat", "pos", "quat", "float", "float"],
        "output_type": "float",
        "default_direction": "decrease",
        "description": "Scale-normalized SE(3) relative pose error: translation and rotation log error.",
    },
    "orientation_error": {
        "input_types": ["quat", "quat"],
        "output_type": "float",
        "default_direction": "decrease",
        "description": "Shortest angular difference in radians between two quaternions.",
    },
    "projection_along_axis": {
        "input_types": ["pos", "pos", "axis"],
        "output_type": "float",
        "default_direction": "increase",
        "description": "Signed projection of point-origin along an axis.",
    },
    "lateral_distance": {
        "input_types": ["pos", "pos", "axis"],
        "output_type": "float",
        "default_direction": "decrease",
        "description": "Distance between two positions perpendicular to an axis.",
    },
    "height_above": {
        "input_types": ["pos", "float"],
        "output_type": "float",
        "default_direction": "increase",
        "description": "Point z coordinate minus a reference height.",
    },
    "absolute_difference": {
        "input_types": ["float", "float"],
        "output_type": "float",
        "default_direction": "decrease",
        "description": "Absolute difference between two scalar values.",
    },
    "vector_norm": {
        "input_types": ["vector"],
        "output_type": "float",
        "default_direction": "decrease",
        "description": "L2 norm of a vector.",
    },
    "cosine_alignment": {
        "input_types": ["axis", "axis"],
        "output_type": "float",
        "default_direction": "increase",
        "description": "Cosine similarity of two axes in [-1, 1].",
    },
    "boolean_state": {
        "input_types": ["bool"],
        "output_type": "float",
        "default_direction": "increase",
        "description": "Convert a boolean task state to 0.0 or 1.0.",
    },
}


def _array(value: Any, expected: int | None = None) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if expected is not None and array.size != expected:
        raise ValueError(f"Expected {expected} values, got {array.size}.")
    if not np.all(np.isfinite(array)):
        raise ValueError("Relation inputs must be finite.")
    return array


def distance(a: Any, b: Any) -> float:
    return float(np.linalg.norm(_array(a, 3) - _array(b, 3)))


def squared_distance(a: Any, b: Any) -> float:
    delta = _array(a, 3) - _array(b, 3)
    return float(delta @ delta)


def orientation_error(a: Any, b: Any) -> float:
    qa = _array(a, 4)
    qb = _array(b, 4)
    qa /= max(float(np.linalg.norm(qa)), 1e-12)
    qb /= max(float(np.linalg.norm(qb)), 1e-12)
    dot = float(np.clip(abs(qa @ qb), 0.0, 1.0))
    return float(2.0 * np.arccos(dot))


def pose_error(a: Any, b: Any, rotation_weight: Any = 0.25) -> float:
    pa = _array(a, 7)
    pb = _array(b, 7)
    weight = float(rotation_weight)
    if not 0.0 <= weight <= 1.0:
        raise ValueError("pose_error rotation_weight must be in [0, 1].")
    return (1.0 - weight) * distance(pa[:3], pb[:3]) + weight * orientation_error(pa[3:], pb[3:])


def relative_pose_norm(
    position: Any,
    quaternion: Any,
    target_position: Any,
    target_quaternion: Any,
    translation_scale: Any = 0.05,
    rotation_scale: Any = 0.25,
) -> float:
    """Return a dimensionless relative-pose error with explicit physical scales.

    Translation is measured in metres and rotation in radians.  The scales are
    therefore required to make the combined norm interpretable rather than an
    arbitrary sum of quantities with different units.
    """
    t_scale = float(translation_scale)
    r_scale = float(rotation_scale)
    if not np.isfinite(t_scale) or not np.isfinite(r_scale) or t_scale <= 0 or r_scale <= 0:
        raise ValueError("relative_pose_norm scales must be finite and positive.")
    translation = distance(position, target_position) / t_scale
    rotation = orientation_error(quaternion, target_quaternion) / r_scale
    return float(np.hypot(translation, rotation))


def _unit_axis(axis: Any) -> np.ndarray:
    value = _array(axis, 3)
    norm = float(np.linalg.norm(value))
    if norm < 1e-12:
        raise ValueError("Axis must be non-zero.")
    return value / norm


def projection_along_axis(point: Any, origin: Any, axis: Any) -> float:
    return float((_array(point, 3) - _array(origin, 3)) @ _unit_axis(axis))


def lateral_distance(point: Any, origin: Any, axis: Any) -> float:
    delta = _array(point, 3) - _array(origin, 3)
    unit = _unit_axis(axis)
    return float(np.linalg.norm(delta - (delta @ unit) * unit))


def height_above(point: Any, reference_height: Any) -> float:
    return float(_array(point, 3)[2] - float(reference_height))


def absolute_difference(a: Any, b: Any) -> float:
    return abs(float(a) - float(b))


def vector_norm(value: Any) -> float:
    return float(np.linalg.norm(_array(value)))


def cosine_alignment(a: Any, b: Any) -> float:
    return float(np.clip(_unit_axis(a) @ _unit_axis(b), -1.0, 1.0))


def boolean_state(value: Any) -> float:
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError("boolean_state expects a boolean value.")
    return float(bool(value))


RELATION_FUNCTIONS: dict[str, Callable[..., float]] = {
    "distance": distance,
    "squared_distance": squared_distance,
    "pose_error": pose_error,
    "relative_pose_norm": relative_pose_norm,
    "orientation_error": orientation_error,
    "projection_along_axis": projection_along_axis,
    "lateral_distance": lateral_distance,
    "height_above": height_above,
    "absolute_difference": absolute_difference,
    "vector_norm": vector_norm,
    "cosine_alignment": cosine_alignment,
    "boolean_state": boolean_state,
}
