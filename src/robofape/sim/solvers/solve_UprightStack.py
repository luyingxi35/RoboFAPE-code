# Copyright (c) 2025 MINT-SJTU
# SPDX-License-Identifier: MIT
# Modified for package-local imports and standalone collection.
import time
import numpy as np
import sapien
from transforms3d.euler import euler2quat
from mani_skill.examples.motionplanning.base_motionplanner.utils import compute_grasp_info_by_obb, get_actor_obb
from mani_skill.examples.motionplanning.panda.motionplanner import PandaArmMotionPlanningSolver
from robofape.sim.tasks.task_UprightStack import UprightStackEnv

def _pose_from_actor(actor):
    return sapien.Pose(p=actor.pose.p[0].detach().cpu().numpy(), q=actor.pose.q[0].detach().cpu().numpy())

def _path_length(result):
    positions = result['position'][:, :7]
    return float(np.linalg.norm(np.diff(positions, axis=0), axis=1).sum())

def _move(planner, pose, rrt_samples=1, max_rrt_length=None):
    if planner.move_to_pose_with_screw(pose) != -1:
        return True
    best_result = None
    best_length = np.inf
    for _ in range(rrt_samples):
        result = planner.move_to_pose_with_RRTConnect(pose, dry_run=True)
        if result == -1:
            continue
        length = _path_length(result)
        if max_rrt_length is not None and length > max_rrt_length:
            continue
        if length < best_length:
            (best_result, best_length) = (result, length)
    if best_result is None:
        return False
    planner.follow_path(best_result)
    return True

def _find_upright_tcp(planner, tcp_pose, target_brick, brick_from_tcp, yaw):
    upright_brick = sapien.Pose(q=euler2quat(0, 0, yaw)) * target_brick
    target_tcp = upright_brick * brick_from_tcp.inv()
    rotate_tcp = sapien.Pose(p=tcp_pose.p, q=target_tcp.q)
    return rotate_tcp

def _align_brick(planner, task, target_p, tolerance):
    """Closed-loop position correction that preserves the selected yaw."""
    for _ in range(18):
        if not bool(task.agent.is_grasping(task.brick).item()):
            return False
        brick_pose = _pose_from_actor(task.brick)
        delta = target_p - brick_pose.p
        if np.linalg.norm(delta) <= tolerance:
            return True
        tcp_pose = _pose_from_actor(task.agent.tcp)
        step = np.clip(delta, -0.03, 0.03)
        if not _move(planner, sapien.Pose(p=tcp_pose.p + step, q=tcp_pose.q), rrt_samples=3, max_rrt_length=3.0):
            return False
    return np.linalg.norm(target_p - _pose_from_actor(task.brick).p) <= tolerance

def _brick_follows_tcp(task, brick_from_tcp, min_lift_z=None):
    tcp_pose = _pose_from_actor(task.agent.tcp)
    brick_pose = _pose_from_actor(task.brick)
    expected_brick = tcp_pose * brick_from_tcp
    position_error = np.linalg.norm(brick_pose.p - expected_brick.p)
    if min_lift_z is not None and brick_pose.p[2] < min_lift_z:
        return False
    return position_error <= 0.06

def _rotate_grasped(planner, task, target_pose, brick_from_tcp):
    if not _move(planner, target_pose, rrt_samples=6, max_rrt_length=3.0):
        return False
    return bool(task.agent.is_grasping(task.brick).item()) and _brick_follows_tcp(task, brick_from_tcp)

def _stage_record(env, stage):
    steps = int(env.unwrapped.elapsed_steps.reshape(-1)[0].item())
    return {'stage': stage, 'end_timestep': max(0, steps - 1), 'elapsed_steps': steps}

def _solve_upright_stack_attempt(env, yaw, seed=None, debug=False, vis=False, reset_options=None):
    env.reset(seed=seed, options=reset_options)
    assert env.unwrapped.control_mode in ('pd_joint_pos', 'pd_joint_pos_vel')
    planner = PandaArmMotionPlanningSolver(env, debug=debug, vis=vis, base_pose=env.unwrapped.agent.robot.pose, visualize_target_grasp_pose=vis, print_env_info=False, joint_vel_limits=0.5, joint_acc_limits=0.5)
    task = env.unwrapped
    stage_records = []
    obb = get_actor_obb(task.brick)
    approaching = np.array([0.0, 0.0, -1.0])
    closing = task.agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
    grasp_info = compute_grasp_info_by_obb(obb, approaching=approaching, target_closing=closing, depth=0.025)
    grasp_pose = task.agent.build_grasp_pose(approaching, grasp_info['closing'], task.brick.pose.sp.p)
    if not _move(planner, grasp_pose * sapien.Pose([0, 0, -0.08])):
        planner.close()
        return -1
    stage_records.append(_stage_record(env, 'approach_pregrasp'))
    if not _move(planner, grasp_pose):
        planner.close()
        return -1
    planner.close_gripper(t=20)
    if not bool(task.agent.is_grasping(task.brick).item()):
        planner.close()
        return -1
    stage_records.append(_stage_record(env, 'grasp'))
    tcp_at_grasp = _pose_from_actor(task.agent.tcp)
    brick_at_grasp = _pose_from_actor(task.brick)
    brick_from_tcp = tcp_at_grasp.inv() * brick_at_grasp
    lift_tcp = sapien.Pose(p=tcp_at_grasp.p + np.array([0.0, 0.0, 0.25]), q=tcp_at_grasp.q)
    if not _move(planner, lift_tcp):
        planner.close()
        return -1
    if not bool(task.agent.is_grasping(task.brick).item()):
        planner.close()
        return -1
    if not _brick_follows_tcp(task, brick_from_tcp, min_lift_z=brick_at_grasp.p[2] + 0.1):
        planner.close()
        return -1
    stage_records.append(_stage_record(env, 'lift'))
    cube_pose = _pose_from_actor(task.cubeA)
    target_brick = cube_pose * sapien.Pose(p=[0.0, 0.0, 0.02 + 0.15 / 2], q=[np.sqrt(0.5), 0.0, np.sqrt(0.5), 0.0])
    rotate_tcp = _find_upright_tcp(planner, lift_tcp, target_brick, brick_from_tcp, yaw)
    if rotate_tcp is None or not _rotate_grasped(planner, task, rotate_tcp, brick_from_tcp):
        planner.close()
        return -1
    if not bool(task.agent.is_grasping(task.brick).item()):
        planner.close()
        return -1
    stage_records.append(_stage_record(env, 'upright'))
    preplace_p = target_brick.p + np.array([0.0, 0.0, 0.1])
    if not _align_brick(planner, task, preplace_p, tolerance=0.01):
        planner.close()
        return -1
    stage_records.append(_stage_record(env, 'pre_place'))
    if not _align_brick(planner, task, target_brick.p, tolerance=0.003):
        planner.close()
        return -1
    planner.open_gripper(t=20)
    time.sleep(0.5)
    tcp_pose = _pose_from_actor(task.agent.tcp)
    retreat = sapien.Pose(p=tcp_pose.p + np.array([0.0, 0.0, 0.2]), q=tcp_pose.q)
    _move(planner, retreat)
    result = planner.open_gripper(t=180)
    success = bool(task.evaluate()['success'].item())
    if success:
        stage_records.append(_stage_record(env, 'final_place'))
    planner.close()
    return (result, stage_records) if success else -1

def _solve_upright_stack(env, seed=None, debug=False, vis=False, reset_options=None):
    for yaw in (-3 * np.pi / 4, -np.pi / 4, -np.pi / 2, np.pi, np.pi / 4, 0, np.pi / 2, 3 * np.pi / 4):
        result = _solve_upright_stack_attempt(env, yaw, seed=seed, debug=debug, vis=vis, reset_options=reset_options)
        if result != -1:
            return result
    return -1

def solveUprightStack(env: UprightStackEnv, seed=None, debug=False, vis=False, reset_options=None):
    return _solve_upright_stack(env, seed=seed, debug=debug, vis=vis, reset_options=reset_options)
