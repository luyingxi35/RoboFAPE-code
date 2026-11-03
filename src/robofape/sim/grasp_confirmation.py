# Copyright (c) 2025 MINT-SJTU
# SPDX-License-Identifier: MIT
# Modified for package-local imports and standalone collection.
"""Shared physical grasp confirmation and trace helpers.

The raw contact values intentionally remain available so labelers can audit the
derived boolean instead of relying on ``agent.is_grasping`` alone.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
import torch


@dataclass(frozen=True)
class GraspThresholds:
    min_force: float = 0.5
    max_angle: float = 85.0
    persistence: int = 3
    loss_persistence: int = 5
    # Panda qpos[-2:].sum() is ~0.046 m for the closed peg grasp in the
    # motion-plan pool; 0.04 incorrectly rejects that valid bilateral grasp.
    # Used only when no object-specific grasp width has been observed yet.
    max_gripper_width: float = 0.04
    open_gripper_width: float = 0.08
    max_follow_residual: float = 0.008
    max_relative_position_drift: float = 0.025
    max_relative_rotation_drift: float = 0.5


@dataclass
class GraspPersistence:
    streak: int = 0
    loss_streak: int = 0
    first_confirmed_frame: int | None = None
    confirmed: bool = False
    lost_frames: list[int] | None = None
    reacquired_frames: list[int] | None = None
    anchor_relative_position: np.ndarray | None = None
    anchor_relative_quaternion: np.ndarray | None = None
    previous_relative_position: np.ndarray | None = None
    previous_relative_quaternion: np.ndarray | None = None
    grasp_width_reference: float | None = None


def classify_contacts(
    left_force: float,
    right_force: float,
    left_angle: float,
    right_angle: float,
    thresholds: GraspThresholds = GraspThresholds(),
) -> Dict[str, Any]:
    left_valid = left_force >= thresholds.min_force and left_angle <= thresholds.max_angle
    right_valid = right_force >= thresholds.min_force and right_angle <= thresholds.max_angle
    return {
        "left_valid": bool(left_valid),
        "right_valid": bool(right_valid),
        "bilateral_valid": bool(left_valid and right_valid),
        "grasp_confirmed_raw": bool(left_valid and right_valid),
    }


def update_grasp(
    state: GraspPersistence,
    frame: int,
    left_force: float,
    right_force: float,
    left_angle: float,
    right_angle: float,
    thresholds: GraspThresholds = GraspThresholds(),
    follow_residual: float | None = None,
    gripper_width: float | None = None,
    relative_position: Any | None = None,
    relative_quaternion: Any | None = None,
) -> Dict[str, Any]:
    result = classify_contacts(left_force, right_force, left_angle, right_angle, thresholds)
    rel_pos = _finite_vector(relative_position, 3)
    rel_quat = _normalized_quaternion(relative_quaternion)
    position_step = _vector_distance(rel_pos, state.previous_relative_position)
    if follow_residual is None:
        follow_residual = position_step
    follow_valid = follow_residual is None or follow_residual <= thresholds.max_follow_residual
    contact_valid = result["grasp_confirmed_raw"]
    if state.grasp_width_reference is None and contact_valid and gripper_width is not None:
        state.grasp_width_reference = float(gripper_width)
    width_limit = thresholds.max_gripper_width
    if state.grasp_width_reference is not None:
        # StackCube rule: midpoint between open width and the observed grasp width.
        width_limit = 0.5 * (float(thresholds.open_gripper_width) + state.grasp_width_reference)
    gripper_closed = gripper_width is None or float(gripper_width) <= width_limit
    position_drift = _vector_distance(rel_pos, state.anchor_relative_position)
    rotation_drift = _quaternion_error(rel_quat, state.anchor_relative_quaternion)
    pose_stable = (
        position_drift is not None
        and rotation_drift is not None
        and position_drift <= thresholds.max_relative_position_drift
        and rotation_drift <= thresholds.max_relative_rotation_drift
    )
    event = None
    if not state.confirmed:
        acquisition_valid = contact_valid and gripper_closed and follow_valid
        state.streak = state.streak + 1 if acquisition_valid else 0
        state.loss_streak = 0
        if state.streak >= thresholds.persistence:
            state.confirmed = True
            state.anchor_relative_position = None if rel_pos is None else rel_pos.copy()
            state.anchor_relative_quaternion = None if rel_quat is None else rel_quat.copy()
            if state.first_confirmed_frame is None:
                state.first_confirmed_frame = frame
                event = "acquired"
            else:
                if state.reacquired_frames is None:
                    state.reacquired_frames = []
                state.reacquired_frames.append(frame)
                event = "reacquired"
    else:
        # Zero contact force is common for a static but still stable grasp. Loss
        # needs independent evidence: opening or relative SE(3) separation.
        if contact_valid and gripper_closed and rel_pos is not None and rel_quat is not None:
            # The object can settle within the fingers during transport/contact.
            # A valid bilateral grasp is authoritative and becomes the new hold
            # anchor; once contact disappears this anchor is frozen for slip tests.
            state.anchor_relative_position = rel_pos.copy()
            state.anchor_relative_quaternion = rel_quat.copy()
            position_drift = 0.0
            rotation_drift = 0.0
            pose_stable = True
        hold_valid = gripper_closed and pose_stable
        loss_evidence = (not gripper_closed) or (not contact_valid and not hold_valid)
        state.loss_streak = state.loss_streak + 1 if loss_evidence else 0
        if state.loss_streak >= thresholds.loss_persistence:
            state.confirmed = False
            state.streak = 0
            state.loss_streak = 0
            state.anchor_relative_position = None
            state.anchor_relative_quaternion = None
            if state.lost_frames is None:
                state.lost_frames = []
            state.lost_frames.append(frame)
            event = "lost"
    state.previous_relative_position = None if rel_pos is None else rel_pos.copy()
    state.previous_relative_quaternion = None if rel_quat is None else rel_quat.copy()
    result.update({
        "grasp_confirmed": state.confirmed,
        "grasp_state": "grasped" if state.confirmed else "candidate",
        "grasp_event": event,
        "acquisition_streak": state.streak,
        "loss_streak": state.loss_streak,
        "gripper_closed": gripper_closed,
        "grasp_width_reference": state.grasp_width_reference,
        "gripper_width_limit": width_limit,
        "follow_residual": follow_residual,
        "follow_valid": follow_valid,
        "relative_position_drift": position_drift,
        "relative_rotation_drift": rotation_drift,
        "kinematic_hold_valid": pose_stable,
    })
    return result


def _finite_vector(value: Any | None, size: int) -> np.ndarray | None:
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    return array if array.size == size and np.all(np.isfinite(array)) else None


def _normalized_quaternion(value: Any | None) -> np.ndarray | None:
    quaternion = _finite_vector(value, 4)
    if quaternion is None:
        return None
    norm = float(np.linalg.norm(quaternion))
    return quaternion / norm if norm > 1e-12 else None


def _vector_distance(a: np.ndarray | None, b: np.ndarray | None) -> float | None:
    return None if a is None or b is None else float(np.linalg.norm(a - b))


def _quaternion_error(a: np.ndarray | None, b: np.ndarray | None) -> float | None:
    if a is None or b is None:
        return None
    return float(2.0 * np.arccos(np.clip(abs(float(a @ b)), 0.0, 1.0)))


def _relative_pose(tcp_position: np.ndarray, tcp_quaternion: np.ndarray,
                   obj_position: np.ndarray, obj_quaternion: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    w, x, y, z = tcp_quaternion
    rotation = np.array([
        [1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
        [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
        [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)],
    ])
    relative_position = rotation.T @ (obj_position - tcp_position)
    iw, ix, iy, iz = (w, -x, -y, -z)
    ow, ox, oy, oz = obj_quaternion
    relative_quaternion = np.array([
        iw*ow - ix*ox - iy*oy - iz*oz,
        iw*ox + ix*ow + iy*oz - iz*oy,
        iw*oy - ix*oz + iy*ow + iz*ox,
        iw*oz + ix*oy - iy*ox + iz*ow,
    ])
    return relative_position, relative_quaternion / np.linalg.norm(relative_quaternion)


def _scalar(value: Any) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().reshape(-1)[0].cpu())
    return float(value)


def observe_grasp(env: Any, obj: Any, thresholds: GraspThresholds = GraspThresholds()) -> Dict[str, Any]:
    """Read one unbatched Panda contact sample directly from the scene."""
    base_env = getattr(env, "unwrapped", env)
    agent = base_env.agent
    scene = base_env.scene
    left_vec = scene.get_pairwise_contact_forces(agent.finger1_link, obj)
    right_vec = scene.get_pairwise_contact_forces(agent.finger2_link, obj)
    left_dir = agent.finger1_link.pose.to_transformation_matrix()[..., :3, 1]
    right_dir = -agent.finger2_link.pose.to_transformation_matrix()[..., :3, 1]
    from mani_skill.utils import common

    left_force = torch.linalg.norm(left_vec, dim=1)
    right_force = torch.linalg.norm(right_vec, dim=1)
    left_angle = torch.rad2deg(common.compute_angle_between(left_dir, left_vec))
    right_angle = torch.rad2deg(common.compute_angle_between(right_dir, right_vec))
    qpos = agent.robot.get_qpos()
    width = torch.abs(qpos[..., -2:]).sum(dim=1)
    tcp = agent.tcp.pose.p
    tcp_q = agent.tcp.pose.q
    obj_pos = obj.pose.p
    obj_q = obj.pose.q
    relative_position, relative_quaternion = _relative_pose(
        tcp[0].detach().cpu().numpy(),
        tcp_q[0].detach().cpu().numpy(),
        obj_pos[0].detach().cpu().numpy(),
        obj_q[0].detach().cpu().numpy(),
    )
    result = {
        "left_ball_force": _scalar(left_force),
        "right_ball_force": _scalar(right_force),
        "left_ball_angle": _scalar(left_angle),
        "right_ball_angle": _scalar(right_angle),
        "gripper_width": _scalar(width),
        "ball_height": _scalar(obj_pos[:, 2]),
        "ball_to_tcp_distance": _scalar(torch.linalg.norm(obj_pos - tcp, dim=1)),
        "agent_is_grasping": bool(agent.is_grasping(obj).reshape(-1)[0].item()),
    }
    # Keep the schema task-neutral; PlaceSphere consumers must never see
    # charger-specific aliases from the PlugCharger collector.
    result["sphere_position"] = obj_pos[0].detach().cpu().tolist()
    result["tcp_position"] = tcp[0].detach().cpu().tolist()
    result["sphere_to_tcp_relative_position"] = relative_position.tolist()
    result["sphere_to_tcp_relative_quaternion"] = relative_quaternion.tolist()
    result["sphere_to_tcp_distance"] = result["ball_to_tcp_distance"]
    result.update(classify_contacts(
        result["left_ball_force"], result["right_ball_force"],
        result["left_ball_angle"], result["right_ball_angle"], thresholds
    ))
    result["left_ball_contact"] = result["left_ball_force"] > 0.0
    result["right_ball_contact"] = result["right_ball_force"] > 0.0
    return result


def summarize_trace(trace: list[Dict[str, Any]], thresholds: GraspThresholds) -> Dict[str, Any]:
    state = GraspPersistence()
    first_lift = None
    baseline = trace[0]["ball_height"] if trace else None
    previous_distance = None
    for frame, row in enumerate(trace):
        follow_residual = row.get("follow_residual")
        if follow_residual is None and previous_distance is not None:
            current_distance = row.get("sphere_to_tcp_distance", row.get("ball_to_tcp_distance"))
            if current_distance is not None:
                follow_residual = abs(float(current_distance) - previous_distance)
        current_distance = row.get("sphere_to_tcp_distance", row.get("ball_to_tcp_distance"))
        if current_distance is not None:
            previous_distance = float(current_distance)
        row.update(update_grasp(
            state, frame, row["left_ball_force"], row["right_ball_force"],
            row["left_ball_angle"], row["right_ball_angle"], thresholds,
            follow_residual=follow_residual,
            gripper_width=row.get("gripper_width"),
            relative_position=row.get("sphere_to_tcp_relative_position"),
            relative_quaternion=row.get("sphere_to_tcp_relative_quaternion"),
        ))
        if baseline is not None and row["ball_height"] >= baseline + 0.04 and first_lift is None:
            first_lift = frame
    return {
        "first_grasp_confirmed_frame": state.first_confirmed_frame,
        "first_lift_confirmed_frame": first_lift,
        "grasp_confirmed": state.confirmed,
        "ever_grasped": state.first_confirmed_frame is not None,
        "grasp_lost_frames": state.lost_frames or [],
        "grasp_reacquired_frames": state.reacquired_frames or [],
        "lift_confirmed": first_lift is not None,
    }
