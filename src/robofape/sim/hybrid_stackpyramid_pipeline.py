# Copyright (c) 2025 MINT-SJTU
# SPDX-License-Identifier: MIT
# Modified for package-local imports and standalone collection.
from __future__ import annotations
import numpy as np

def metric(row, stage, ep):
    tcp = np.asarray(row['tcp'])
    a = np.asarray(row['cubeA'])
    b = np.asarray(row['cubeB'])
    c = np.asarray(row['cubeC'])
    if stage == 0:
        return float(np.linalg.norm(tcp - ep['pregrasp_A']))
    if stage == 1:
        return abs(float(row['gripper_width']) - ep['grasp_width_A'])
    if stage == 2:
        return float(np.linalg.norm((a - b)[:2]))
    if stage == 3:
        return float(np.linalg.norm(tcp - ep['pregrasp_C']))
    if stage == 4:
        return abs(float(row['gripper_width']) - ep['grasp_width_C'])
    if stage == 5:
        return abs(float(c[2]) - ep['lift_z'])
    if stage == 6:
        return float(np.linalg.norm(c - ep['pre_release']))
    return float(np.linalg.norm(c - ep['final_target']))

def classify(row, ep):
    a = np.asarray(row['cubeA'])
    b = np.asarray(row['cubeB'])
    c = np.asarray(row['cubeC'])
    base = float(np.linalg.norm((a - b)[:2])) <= ep['base_tol']
    final = float(np.linalg.norm(c - ep['final_target'])) <= 0.03 and (not row['grasp_C_raw']) and row.get('cubeC_static', True)
    if final:
        return 7
    if float(np.linalg.norm(c - ep['pre_release'])) <= 0.04 and row['grasp_C_raw']:
        return 6
    if row['grasp_C_raw'] and c[2] >= ep['initial_C_z'] + 0.1:
        return 5
    if row['grasp_C_raw']:
        return 4
    if base and float(np.linalg.norm(np.asarray(row['tcp']) - np.asarray(ep['pregrasp_C']))) <= 0.08:
        return 3
    if base and (not row['grasp_A_raw']):
        return 2
    if row['grasp_A_raw']:
        return 1
    return 0

def endpoints(rows):
    r0 = rows[0]
    a = np.asarray(r0['cubeA'])
    b = np.asarray(r0['cubeB'])
    c = np.asarray(r0['cubeC'])
    return {'pregrasp_A': np.asarray(r0['tcp']).tolist(), 'pregrasp_C': np.asarray(r0['tcp']).tolist(), 'grasp_width_A': 0.04, 'grasp_width_C': 0.04, 'initial_C_z': float(c[2]), 'lift_z': float(c[2] + 0.1), 'base_tol': float(np.sqrt(2) * 0.02 + 0.005), 'pre_release': ((a + b) / 2 + np.array([0, 0, 0.04])).tolist(), 'final_target': ((a + b) / 2 + np.array([0, 0, 0.04])).tolist()}
