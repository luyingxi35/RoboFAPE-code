# Copyright (c) 2025 MINT-SJTU
# SPDX-License-Identifier: MIT
# Modified for package-local imports and standalone collection.
from __future__ import annotations
import numpy as np

def _vec(row, key):
    value = np.asarray(row.get(key, [np.nan, np.nan, np.nan]), dtype=float).reshape(-1)
    if value.size < 3:
        return np.full(3, np.nan)
    return value[:3]

def classify(trace, lift_delta=0.04, target_radius=0.05):
    if not trace:
        return ([], np.zeros(0), [])
    base_z = _vec(trace[0], 'source_obj_pos')[2]
    (stages, metrics, names) = ([], [], [])
    for row in trace:
        (src, target, eef) = (_vec(row, 'source_obj_pos'), _vec(row, 'target_obj_pos'), _vec(row, 'eef_pos'))
        grasped = bool(row.get('source_obj_grasped', row.get('grasp_confirmed', False)))
        target_distance = float(np.linalg.norm(src - target)) if np.all(np.isfinite(src)) and np.all(np.isfinite(target)) else np.inf
        lifted = bool(np.isfinite(src[2]) and np.isfinite(base_z) and (src[2] >= base_z + lift_delta))
        if target_distance <= target_radius:
            (stage, metric, name) = ('place_release', target_distance, 'source_target_distance')
        elif grasped and lifted:
            (stage, metric, name) = ('transport', target_distance, 'source_target_distance')
        elif grasped:
            (stage, metric, name) = ('grasp_lift', max(0.0, base_z + lift_delta - src[2]), 'lift_height_gap')
        else:
            (stage, metric, name) = ('approach', float(np.linalg.norm(eef - src)), 'tcp_source_distance')
        stages.append(stage)
        metrics.append(metric)
        names.append(name)
    return (stages, np.asarray(metrics, dtype=float), names)
