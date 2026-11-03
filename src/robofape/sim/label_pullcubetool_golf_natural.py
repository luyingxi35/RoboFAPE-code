# Copyright (c) 2025 MINT-SJTU
# SPDX-License-Identifier: MIT
# Modified for package-local imports and standalone collection.
import numpy as np
(VELOCITY_THRESHOLD, VELOCITY_PERSISTENCE) = (0.01, 3)

def _arr(trace, key, shape=(3,)):
    values = np.asarray([row.get(key, [np.nan] * shape[0]) for row in trace], dtype=float)
    return values

def classify_trace(trace):
    grasp = np.asarray([bool(row.get('tool_grasp_confirmed', False)) for row in trace])
    velocity = np.asarray([float(row.get('ball_forward_velocity', np.nan)) for row in trace])
    success = np.asarray([bool(row.get('pull_success', row.get('success', False))) for row in trace])
    active = np.zeros(len(trace), dtype=bool)
    streak = 0
    for (i, value) in enumerate(velocity >= VELOCITY_THRESHOLD):
        streak = streak + 1 if value else 0
        active[i] = streak >= VELOCITY_PERSISTENCE
    stages = np.full(len(trace), 'acquire_tool', dtype=object)
    current = 'acquire_tool'
    for i in range(len(trace)):
        if current != 'pull_ball' and (active[i] or success[i]):
            current = 'pull_ball'
        elif current == 'acquire_tool' and grasp[i]:
            current = 'transport_tool'
        stages[i] = current
    return stages.tolist()

def _metric(trace, stage):
    (eef, tool, ball) = (_arr(trace, 'eef_pos'), _arr(trace, 'tool_pos'), _arr(trace, 'cube_pos'))
    if stage == 'acquire_tool':
        return np.linalg.norm(eef - tool, axis=1)
    if stage == 'transport_tool':
        return np.linalg.norm(tool - ball, axis=1)
    return np.linalg.norm(ball[:, :2], axis=1)
