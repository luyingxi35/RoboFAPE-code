# Copyright (c) 2025 MINT-SJTU
# SPDX-License-Identifier: MIT
# Modified for package-local imports and standalone collection.
import numpy as np
import sapien
from mani_skill.examples.motionplanning.panda.motionplanner import PandaArmMotionPlanningSolver
from mani_skill.examples.motionplanning.base_motionplanner.utils import compute_grasp_info_by_obb, get_actor_obb
from robofape.sim.tasks.task_PickCube import PickCubeEnv_ball

def solve_pickcube_ball(env: PickCubeEnv_ball, seed=None, debug=False, vis=False):
    env.reset(seed=seed)
    planner = PandaArmMotionPlanningSolver(env, debug=debug, vis=vis, base_pose=env.unwrapped.agent.robot.pose, visualize_target_grasp_pose=vis, print_env_info=False)
    FINGER_LENGTH = 0.025
    env = env.unwrapped
    obb = get_actor_obb(env.cube)
    approaching = np.array([0, 0, -1])
    target_closing = env.agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
    grasp_info = compute_grasp_info_by_obb(obb, approaching=approaching, target_closing=target_closing, depth=FINGER_LENGTH)
    (closing, center) = (grasp_info['closing'], grasp_info['center'])
    grasp_pose = env.agent.build_grasp_pose(approaching, closing, env.cube.pose.sp.p)
    reach_pose = grasp_pose * sapien.Pose([0, 0, -0.05])
    planner.move_to_pose_with_screw(reach_pose)
    planner.move_to_pose_with_screw(grasp_pose)
    planner.close_gripper()
    goal_pose = sapien.Pose(env.goal_site.pose.sp.p, grasp_pose.q)
    res = planner.move_to_pose_with_screw(goal_pose)
    planner.close()
    return res
