# Copyright (c) 2025 MINT-SJTU
# SPDX-License-Identifier: MIT
# Modified for package-local imports and standalone collection.
from __future__ import annotations
from dataclasses import dataclass
from enum import IntEnum
import math
from typing import Any
import numpy as np

class UprightStage(IntEnum):
    APPROACH_PREGRASP = 1
    GRASP = 2
    LIFT = 3
    UPRIGHT = 4
    PRE_PLACE = 5
    FINAL_PLACE = 6

@dataclass(frozen=True)
class StageEndpoints:
    pregrasp_tcp: np.ndarray
    grasp_width: float
    initial_brick: np.ndarray
    lift_height: float
    upright_axis: np.ndarray
    pre_place_brick: np.ndarray
    final_brick: np.ndarray

@dataclass(frozen=True)
class UprightState:
    tcp_position: np.ndarray
    gripper_width: float
    brick_position: np.ndarray
    brick_axis: np.ndarray
    is_grasping: bool
    cube_a_static: bool
    cube_b_static: bool
    brick_static: bool
    brick_on_target: bool = False

@dataclass(frozen=True)
class StageFrame:
    stage: UprightStage
    metric_name: str
    metric_value: float
    gate: str

def _vec(value: Any, size: int=3) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64).reshape(-1)
    if result.size != size:
        raise ValueError(f'expected vector of size {size}, got {result.size}')
    return result

def _angle(a: np.ndarray, b: np.ndarray) -> float:
    a = _vec(a)
    b = _vec(b)
    (na, nb) = (np.linalg.norm(a), np.linalg.norm(b))
    if na == 0 or nb == 0:
        return math.pi
    cosine = float(np.clip(abs(np.dot(a, b) / (na * nb)), -1.0, 1.0))
    return math.acos(cosine)

def classify_state(state: UprightState, endpoints: StageEndpoints) -> UprightStage:
    """Map every state to exactly one stage using the documented priority tree."""
    tcp_brick_distance = float(np.linalg.norm(_vec(state.tcp_position) - _vec(state.brick_position)))
    pregrasp_distance = float(np.linalg.norm(_vec(endpoints.pregrasp_tcp) - _vec(endpoints.initial_brick)))
    approached = tcp_brick_distance <= pregrasp_distance + 0.005
    closed = state.gripper_width <= (0.08 + float(endpoints.grasp_width)) / 2.0
    grasped = state.is_grasping or (closed and tcp_brick_distance <= 0.02)
    lift = float(state.brick_position[2] - endpoints.initial_brick[2]) >= 0.1
    upright = abs(float(np.dot(_vec(state.brick_axis), np.array([0.0, 0.0, 1.0])))) / max(np.linalg.norm(state.brick_axis), 1e-12) >= 0.95
    final = float(np.linalg.norm(_vec(state.brick_position) - _vec(endpoints.final_brick))) <= 0.03 and upright and (state.gripper_width >= 0.06)
    if final:
        return UprightStage.FINAL_PLACE
    if grasped and upright:
        return UprightStage.PRE_PLACE
    if grasped and lift:
        return UprightStage.UPRIGHT
    if grasped:
        return UprightStage.LIFT
    if approached:
        return UprightStage.GRASP
    return UprightStage.APPROACH_PREGRASP

def metric_for_state(stage: UprightStage, state: UprightState, endpoints: StageEndpoints) -> tuple[str, float]:
    """Return one scalar distance for the active stage; gates never get summed."""
    if stage is UprightStage.APPROACH_PREGRASP:
        return ('tcp_pregrasp_distance', float(np.linalg.norm(state.tcp_position - endpoints.pregrasp_tcp)))
    if stage is UprightStage.GRASP:
        return ('gripper_width_distance', abs(float(state.gripper_width) - float(endpoints.grasp_width)))
    if stage is UprightStage.LIFT:
        return ('brick_height_distance', abs(float(state.brick_position[2]) - float(endpoints.lift_height)))
    if stage is UprightStage.UPRIGHT:
        return ('upright_angle_distance', _angle(state.brick_axis, endpoints.upright_axis))
    if stage is UprightStage.PRE_PLACE:
        return ('pre_place_position_distance', float(np.linalg.norm(state.brick_position - endpoints.pre_place_brick)))
    return ('final_place_position_distance', float(np.linalg.norm(state.brick_position - endpoints.final_brick)))

def label_state(state: UprightState, endpoints: StageEndpoints) -> StageFrame:
    stage = classify_state(state, endpoints)
    (metric_name, metric_value) = metric_for_state(stage, state, endpoints)
    return StageFrame(stage, metric_name, metric_value, stage.name.lower())
