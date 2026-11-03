# Copyright (c) 2025 MINT-SJTU
# SPDX-License-Identifier: MIT
# Modified for package-local imports and standalone collection.
import numpy as np
import sapien
from mani_skill.examples.motionplanning.panda.motionplanner import PandaArmMotionPlanningSolver
from robofape.sim.tasks.task_PullCube import PullCubeEnv_block

def solve_pullcube_block(env: PullCubeEnv_block, seed=None, debug=False, vis=False):
    env.reset(seed=seed)
    planner = PandaArmMotionPlanningSolver(env, debug=debug, vis=vis, base_pose=env.unwrapped.agent.robot.pose, visualize_target_grasp_pose=vis, print_env_info=False)
    FINGER_LENGTH = 0.025
    env = env.unwrapped
    planner.close_gripper()
    reach_pose = sapien.Pose(p=env.obj.pose.sp.p + np.array([0.08, 0, 0.03]), q=env.agent.tcp.pose.sp.q)
    planner.move_to_pose_with_screw(reach_pose)
    goal_pose = sapien.Pose(p=env.goal_region.pose.sp.p + np.array([0.05, 0, 0]), q=env.agent.tcp.pose.sp.q)
    res = planner.move_to_pose_with_screw(goal_pose)
    planner.close()
    return res
