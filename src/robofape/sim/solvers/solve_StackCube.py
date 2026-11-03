# Copyright (c) 2025 MINT-SJTU
# SPDX-License-Identifier: MIT
# Modified for package-local imports and standalone collection.
import numpy as np
import sapien
import time
from transforms3d.euler import euler2quat
from mani_skill.examples.motionplanning.panda.motionplanner import PandaArmMotionPlanningSolver
from mani_skill.examples.motionplanning.base_motionplanner.utils import compute_grasp_info_by_obb, get_actor_obb
from robofape.sim.tasks.task_StackCube import StackCubeEnv_can, StackCubeEnv_orangebowl, StackCubeEnv_dicebrick

def _to_numpy(value):
    if hasattr(value, 'detach'):
        return value.detach().cpu().numpy()
    return np.asarray(value)

def solve_stackcube_can(env: StackCubeEnv_can, seed=None, debug=False, vis=False):
    env.reset(seed=seed)
    assert env.unwrapped.control_mode in ['pd_joint_pos', 'pd_joint_pos_vel'], env.unwrapped.control_mode
    planner = PandaArmMotionPlanningSolver(env, debug=debug, vis=vis, base_pose=env.unwrapped.agent.robot.pose, visualize_target_grasp_pose=vis, print_env_info=False)
    FINGER_LENGTH = 0.025
    env = env.unwrapped
    obb = get_actor_obb(env.cubeA)
    approaching = np.array([0, 0, -1])
    target_closing = _to_numpy(env.agent.tcp.pose.to_transformation_matrix()[0, :3, 1])
    grasp_info = compute_grasp_info_by_obb(obb, approaching=approaching, target_closing=target_closing, depth=FINGER_LENGTH)
    (closing, center) = (grasp_info['closing'], env.cubeA.pose.sp.p)
    grasp_pose = env.agent.build_grasp_pose(approaching, closing, center)
    offset = np.array([0.0, 0.0, -0.02])
    grasp_pose_offset = grasp_pose * sapien.Pose(offset)
    delta_angle = 0
    delta_quat = euler2quat(0, 0, delta_angle)
    delta_pose = sapien.Pose(q=delta_quat)
    grasp_pose_offset = grasp_pose_offset * delta_pose
    angles = np.arange(0, np.pi * 2 / 3, np.pi / 2)
    angles = np.repeat(angles, 2)
    angles[1::2] *= -1
    for angle in angles:
        delta_pose = sapien.Pose(q=euler2quat(0, 0, angle))
        grasp_pose2 = grasp_pose_offset * delta_pose
        res = planner.move_to_pose_with_screw(grasp_pose2, dry_run=True)
        if res == -1:
            continue
        grasp_pose = grasp_pose2
        break
    else:
        print('Fail to find a valid grasp pose')
    reach_pose = grasp_pose * sapien.Pose([0, 0, -0.05])
    planner.move_to_pose_with_screw(reach_pose)
    planner.move_to_pose_with_screw(grasp_pose)
    planner.close_gripper()
    lift_pose = sapien.Pose([0, 0, 0.15]) * grasp_pose
    planner.move_to_pose_with_screw(lift_pose)
    goal_pose = env.cubeB.pose * sapien.Pose([0, 0, float(env.cube_half_size[2].item() * 4)])
    offset = _to_numpy(goal_pose.p - env.cubeA.pose.p)[0]
    align_pose = sapien.Pose(lift_pose.p + offset, lift_pose.q)
    planner.move_to_pose_with_screw(align_pose)
    res = planner.open_gripper()
    time.sleep(0.5)
    planner.close()
    return res

def solve_stackcube_orangebowl(env: StackCubeEnv_orangebowl, seed=None, debug=False, vis=False):
    env.reset(seed=seed)
    assert env.unwrapped.control_mode in ['pd_joint_pos', 'pd_joint_pos_vel'], env.unwrapped.control_mode
    planner = PandaArmMotionPlanningSolver(env, debug=debug, vis=vis, base_pose=env.unwrapped.agent.robot.pose, visualize_target_grasp_pose=vis, print_env_info=False)
    FINGER_LENGTH = 0.025
    env = env.unwrapped
    obb = get_actor_obb(env.cubeA)
    approaching = np.array([0, 0, -1])
    target_closing = _to_numpy(env.agent.tcp.pose.to_transformation_matrix()[0, :3, 1])
    grasp_info = compute_grasp_info_by_obb(obb, approaching=approaching, target_closing=target_closing, depth=FINGER_LENGTH)
    (closing, center) = (grasp_info['closing'], env.cubeA.pose.sp.p)
    grasp_pose = env.agent.build_grasp_pose(approaching, closing, center)
    delta_angle = 0
    delta_quat = euler2quat(0, 0, delta_angle)
    delta_pose = sapien.Pose(q=delta_quat)
    grasp_pose_offset = grasp_pose * delta_pose
    angles = np.arange(0, np.pi * 2 / 3, np.pi / 2)
    angles = np.repeat(angles, 2)
    angles[1::2] *= -1
    for angle in angles:
        delta_pose = sapien.Pose(q=euler2quat(0, 0, angle))
        grasp_pose2 = grasp_pose_offset * delta_pose
        res = planner.move_to_pose_with_screw(grasp_pose2, dry_run=True)
        if res == -1:
            continue
        grasp_pose = grasp_pose2
        break
    else:
        print('Fail to find a valid grasp pose')
    reach_pose = grasp_pose * sapien.Pose([0, 0, -0.05])
    planner.move_to_pose_with_screw(reach_pose)
    planner.move_to_pose_with_screw(grasp_pose)
    planner.close_gripper()
    lift_pose = sapien.Pose([0, 0, 0.15]) * grasp_pose
    planner.move_to_pose_with_screw(lift_pose)
    goal_pose = env.cubeB.pose * sapien.Pose([0, 0, float(env.cube_half_size[2].item() * 4)])
    offset = _to_numpy(goal_pose.p - env.cubeA.pose.p)[0]
    align_pose = sapien.Pose(lift_pose.p + offset, lift_pose.q)
    planner.move_to_pose_with_screw(align_pose)
    res = planner.open_gripper()
    time.sleep(0.5)
    planner.close()
    return res

def solve_stackcube_dicebrick(env: StackCubeEnv_dicebrick, seed=None, debug=False, vis=False):
    env.reset(seed=seed)
    assert env.unwrapped.control_mode in ['pd_joint_pos', 'pd_joint_pos_vel'], env.unwrapped.control_mode
    planner = PandaArmMotionPlanningSolver(env, debug=debug, vis=vis, base_pose=env.unwrapped.agent.robot.pose, visualize_target_grasp_pose=vis, print_env_info=False)
    FINGER_LENGTH = 0.025
    env = env.unwrapped
    obb = get_actor_obb(env.cubeA)
    approaching = np.array([0, 0, -1])
    target_closing = _to_numpy(env.agent.tcp.pose.to_transformation_matrix()[0, :3, 1])
    grasp_info = compute_grasp_info_by_obb(obb, approaching=approaching, target_closing=target_closing, depth=FINGER_LENGTH)
    (closing, center) = (grasp_info['closing'], env.cubeA.pose.sp.p)
    grasp_pose = env.agent.build_grasp_pose(approaching, closing, center)
    offset = np.array([0.0, 0.0, 0.0])
    grasp_pose_offset = grasp_pose * sapien.Pose(offset)
    delta_angle = 0
    delta_quat = euler2quat(0, 0, delta_angle)
    delta_pose = sapien.Pose(q=delta_quat)
    grasp_pose_offset = grasp_pose_offset * delta_pose
    angles = np.arange(0, np.pi * 2 / 3, np.pi / 2)
    angles = np.repeat(angles, 2)
    angles[1::2] *= -1
    for angle in angles:
        delta_pose = sapien.Pose(q=euler2quat(0, 0, angle))
        grasp_pose2 = grasp_pose_offset * delta_pose
        res = planner.move_to_pose_with_screw(grasp_pose2, dry_run=True)
        if res == -1:
            continue
        grasp_pose = grasp_pose2
        break
    else:
        print('Fail to find a valid grasp pose')
    reach_pose = grasp_pose * sapien.Pose([0, 0, -0.05])
    planner.move_to_pose_with_screw(reach_pose)
    planner.move_to_pose_with_screw(grasp_pose)
    planner.close_gripper()
    lift_pose = sapien.Pose([0, 0, 0.15]) * grasp_pose
    planner.move_to_pose_with_screw(lift_pose)
    goal_pose = env.cubeB.pose * sapien.Pose([0, 0, float(env.cube_half_size[2].item() * 2.5)])
    offset = _to_numpy(goal_pose.p - env.cubeA.pose.p)[0]
    align_pose = sapien.Pose(lift_pose.p + offset, lift_pose.q)
    planner.move_to_pose_with_screw(align_pose)
    res = planner.open_gripper()
    time.sleep(0.5)
    planner.close()
    return res
