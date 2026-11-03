# Copyright (c) 2025 MINT-SJTU
# SPDX-License-Identifier: MIT
# Modified for package-local imports and standalone collection.
from __future__ import annotations
import numpy as np
PULL_START_METERS = 0.01
PULL_POINT_OFFSET = np.array([0.03, 0.0, 0.0], dtype=np.float64)

def phase_metrics(initial_obj, goal, steps):
    axis = np.asarray(goal[:2] - initial_obj[:2], dtype=np.float64)
    norm = float(np.linalg.norm(axis))
    if norm <= 1e-08:
        raise ValueError('initial object and goal positions coincide')
    axis /= norm
    (stages, approach, goal_distance, displacement) = ([], [], [], [])
    for step in steps:
        (obj, eef) = (step['obj_pos'], step['eef_pos'])
        moved = float(np.dot(obj[:2] - initial_obj[:2], axis))
        displacement.append(moved)
        approach.append(float(np.linalg.norm(eef - (obj + PULL_POINT_OFFSET))))
        goal_distance.append(float(np.linalg.norm(obj[:2] - goal[:2])))
        stages.append('pull_to_goal' if moved >= PULL_START_METERS else 'approach_pull_point')
    return (stages, approach, goal_distance, displacement)
