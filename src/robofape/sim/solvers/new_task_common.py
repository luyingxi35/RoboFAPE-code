# Copyright (c) 2025 MINT-SJTU
# SPDX-License-Identifier: MIT
# Modified for package-local imports and standalone collection.
"""Small, shared motion-planning helpers for the new Panda wrist-camera tasks."""

from pathlib import Path

import numpy as np
import sapien

from mani_skill.examples.motionplanning.panda.motionplanner import (
    PandaArmMotionPlanningSolver,
)


def actor_pose(actor):
    """Return the single-environment SAPIEN pose without changing the actor."""
    return actor.pose.sp


def wrist_planner(env, *, debug=False, vis=False):
    base = env.unwrapped
    if base.num_envs != 1:
        raise ValueError("Motion-planning solvers require num_envs=1")
    if base.robot_uids != "panda_wristcam":
        raise ValueError(
            f"Expected panda_wristcam, got {base.robot_uids!r}; "
            "create the environment with the wrist-camera robot"
        )
    if base.control_mode != "pd_joint_pos":
        raise ValueError("Panda motion planning requires control_mode='pd_joint_pos'")
    urdf = Path(base.agent.urdf_path)
    srdf = urdf.with_suffix(".srdf")
    if not urdf.is_file() or not srdf.is_file():
        raise FileNotFoundError(
            f"Wrist-camera robot planning assets missing: {urdf}, {srdf}"
        )
    return PandaArmMotionPlanningSolver(
        env,
        debug=debug,
        vis=vis,
        base_pose=base.agent.robot.pose,
        visualize_target_grasp_pose=vis,
        print_env_info=False,
    )


def move(planner, pose, *, allow_rrt=False, refine_steps=0):
    result = planner.move_to_pose_with_screw(pose, refine_steps=refine_steps)
    if result == -1 and allow_rrt:
        result = planner.move_to_pose_with_RRTConnect(pose, refine_steps=refine_steps)
    return result


def translated(pose, delta):
    return sapien.Pose(np.asarray(pose.p) + np.asarray(delta), pose.q)


def target_tcp_for_object(obj_pose, tcp_pose, target_obj_pose):
    """Preserve the measured object-to-TCP transform after the grasp."""
    return target_obj_pose * obj_pose.inv() * tcp_pose


def top_grasp_pose(env, obj, *, depth=0.025, closing_axis=None, root_offset=False):
    """Build a centered top-down grasp from the object's world OBB."""
    from mani_skill.examples.motionplanning.base_motionplanner.utils import (
        compute_grasp_info_by_obb,
        get_actor_obb,
    )

    approaching = np.array([0.0, 0.0, -1.0])
    if closing_axis is None:
        target_closing = (
            env.agent.tcp.pose.to_transformation_matrix()[0, :3, 1]
            .cpu()
            .numpy()
        )
    else:
        target_closing = np.asarray(closing_axis, dtype=np.float32)
    obb = get_actor_obb(obj)
    info = compute_grasp_info_by_obb(
        obb,
        approaching=approaching,
        target_closing=target_closing,
        depth=depth,
    )
    if root_offset:
        # Some actors have a geometry origin offset from their root pose.
        root_transform = np.asarray(obj.pose.sp.to_transformation_matrix())
        center = (root_transform @ np.append(info["center"], 1.0))[:3]
        closing = root_transform[:3, :3] @ info["closing"]
        closing /= np.linalg.norm(closing)
    else:
        center, closing = info["center"], info["closing"]
    return env.agent.build_grasp_pose(approaching, closing, center)


def settle(planner, steps=6):
    """Hold the current arm pose so static-success conditions can be evaluated."""
    qpos = (
        planner.robot.get_qpos()[0, : len(planner.planner.joint_vel_limits)]
        .cpu()
        .numpy()
    )
    result = None
    for _ in range(steps):
        result = planner.env.step(np.hstack([qpos, planner.gripper_state]))
    return result
