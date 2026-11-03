# Copyright (c) 2025 MINT-SJTU
# SPDX-License-Identifier: MIT
# Modified for package-local imports and standalone collection.
from mani_skill.envs.tasks.tabletop.poke_cube import PokeCubeEnv
from mani_skill.utils.registration import register_env

@register_env('PokeCubeWristCam-v1', max_episode_steps=120)
class PokeCubeWristCamEnv(PokeCubeEnv):
    SUPPORTED_ROBOTS = ['panda_wristcam']

    def __init__(self, *args, robot_uids='panda_wristcam', **kwargs):
        super().__init__(*args, robot_uids=robot_uids, **kwargs)
