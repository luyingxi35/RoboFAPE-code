# Copyright (c) 2025 MINT-SJTU
# SPDX-License-Identifier: MIT
# Modified for package-local imports and standalone collection.
import numpy as np
import sapien
from mani_skill.envs.tasks import PegInsertionSideEnv
from mani_skill.examples.motionplanning.panda.motionplanner import PandaArmMotionPlanningSolver
from mani_skill.examples.motionplanning.base_motionplanner.utils import get_actor_obb
PREINSERT_Z_BIAS = 0.004

def solve_peginsertionside(env: PegInsertionSideEnv, seed=None, debug=False, vis=False, reset_options=None):
    """Motion plan using current simulator poses and closed-loop alignment."""
    env.reset(seed=seed, options=reset_options)
    assert env.unwrapped.control_mode in ['pd_joint_pos', 'pd_joint_pos_vel'], env.unwrapped.control_mode
    planner = PandaArmMotionPlanningSolver(env, debug=debug, vis=vis, base_pose=env.unwrapped.agent.robot.pose, visualize_target_grasp_pose=vis, print_env_info=False, joint_vel_limits=0.75, joint_acc_limits=0.75)
    task = env.unwrapped
    obb = get_actor_obb(task.peg)
    approaching = np.array([0.0, 0.0, -1.0])
    target_closing = task.agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
    peg_initial_pose = task.peg.pose.sp
    peg_rotation = task.peg.pose.to_transformation_matrix()[0, :3, :3].cpu().numpy()
    closing = peg_rotation[:, 1]
    if float(np.dot(closing, target_closing)) < 0:
        closing = -closing
    half_length = float(task.peg_half_sizes[0, 0].item())
    tail_pose = peg_initial_pose * sapien.Pose([-max(0.05, half_length / 2 + 0.01), 0.0, 0.0])
    grasp_center = np.asarray(tail_pose.p, dtype=np.float64)
    grasp_pose = task.agent.build_grasp_pose(approaching, closing, grasp_center)
    if planner.move_to_pose_with_screw(grasp_pose * sapien.Pose([0.0, 0.0, -0.05])) == -1:
        planner.close()
        return -1
    if planner.move_to_pose_with_screw(grasp_pose) == -1:
        planner.close()
        return -1
    planner.close_gripper()
    preinsert_offset = sapien.Pose([-0.01 - half_length, 0.0, PREINSERT_Z_BIAS])
    desired_preinsert_peg_pose = task.goal_pose.sp * preinsert_offset
    preinsert_pose = desired_preinsert_peg_pose * task.peg.pose.sp.inv() * task.agent.tcp.pose.sp
    if planner.move_to_pose_with_screw(preinsert_pose) == -1:
        planner.close()
        return -1
    for _ in range(3):
        preinsert_pose = desired_preinsert_peg_pose * task.peg.pose.sp.inv() * task.agent.tcp.pose.sp
        if planner.move_to_pose_with_screw(preinsert_pose) == -1:
            planner.close()
            return -1
    desired_final_peg_pose = task.goal_pose.sp * sapien.Pose([0.05, 0.0, PREINSERT_Z_BIAS])
    final_pose = desired_final_peg_pose * task.peg.pose.sp.inv() * task.agent.tcp.pose.sp
    res = planner.move_to_pose_with_screw(final_pose)
    planner.close()
    return res
