# Copyright (c) 2025 MINT-SJTU
# SPDX-License-Identifier: MIT
# Modified for package-local imports and standalone collection.
from __future__ import annotations
from dataclasses import dataclass
from enum import IntEnum
import numpy as np

class PlaceSphereStage(IntEnum):
    APPROACH = 1
    GRASP = 2
    LIFT = 3
    TRANSPORT_ALIGN = 4
    RELEASE_SETTLE = 5
TAU_LIFT_START = 0.02
TAU_LIFT_DONE = 0.01
TAU_TARGET = 0.02

@dataclass(frozen=True)
class PlaceSphereEndpoints:
    initial_sphere: np.ndarray
    grasp_width: float
    lift_z: float
    final_sphere: np.ndarray
    target: np.ndarray

@dataclass(frozen=True)
class StageFrame:
    stage: PlaceSphereStage
    metric_name: str
    metric_value: float

@dataclass(frozen=True)
class PlaceSphereState:
    tcp: np.ndarray
    sphere: np.ndarray
    width: float
    grasp_confirmed: bool
    success: bool

def classify_state(state: PlaceSphereState, endpoints: PlaceSphereEndpoints) -> PlaceSphereStage:
    sphere = np.asarray(state.sphere, dtype=float)
    target_distance = float(np.linalg.norm(sphere - endpoints.target))
    dz = float(sphere[2] - endpoints.initial_sphere[2])
    lift_done = abs(float(sphere[2]) - float(endpoints.lift_z)) <= TAU_LIFT_DONE
    if state.success:
        return PlaceSphereStage.RELEASE_SETTLE
    if target_distance <= TAU_TARGET:
        return PlaceSphereStage.TRANSPORT_ALIGN
    if state.grasp_confirmed and lift_done:
        return PlaceSphereStage.TRANSPORT_ALIGN
    if state.grasp_confirmed and dz >= TAU_LIFT_START:
        return PlaceSphereStage.LIFT
    if state.grasp_confirmed:
        return PlaceSphereStage.GRASP
    return PlaceSphereStage.APPROACH

def metric_for_state(stage: PlaceSphereStage, state: PlaceSphereState, endpoints: PlaceSphereEndpoints) -> tuple[str, float]:
    if stage is PlaceSphereStage.APPROACH:
        return ('tcp_sphere_distance', float(np.linalg.norm(state.tcp - state.sphere)))
    if stage is PlaceSphereStage.GRASP:
        return ('gripper_width_distance', abs(float(state.width) - endpoints.grasp_width))
    if stage is PlaceSphereStage.LIFT:
        return ('sphere_lift_height_distance', abs(float(state.sphere[2]) - endpoints.lift_z))
    if stage is PlaceSphereStage.TRANSPORT_ALIGN:
        return ('sphere_target_distance', float(np.linalg.norm(state.sphere - endpoints.target)))
    return ('sphere_final_distance', float(np.linalg.norm(state.sphere - endpoints.final_sphere)))

def label_state(state: PlaceSphereState, endpoints: PlaceSphereEndpoints) -> StageFrame:
    stage = classify_state(state, endpoints)
    (name, value) = metric_for_state(stage, state, endpoints)
    return StageFrame(stage, name, value)
