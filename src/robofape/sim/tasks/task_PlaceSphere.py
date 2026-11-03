# Copyright (c) 2025 MINT-SJTU
# SPDX-License-Identifier: MIT
# Modified for package-local imports and standalone collection.
from typing import Union
from mani_skill.agents.robots import Fetch, Panda, PandaWristCam
from mani_skill.envs.tasks.tabletop.place_sphere import PlaceSphereEnv as _BasePlaceSphereEnv
from mani_skill.utils.registration import register_env

def _with_wrist_camera_sensor_config(sensor_configs):
    merged = dict(sensor_configs or {})
    hand = dict(merged.get('hand_camera', {}))
    hand.setdefault('width', 224)
    hand.setdefault('height', 224)
    merged['hand_camera'] = hand
    return merged

@register_env('PlaceSphere-v1', max_episode_steps=50, override=True)
class PlaceSphereWristCamEnv(_BasePlaceSphereEnv):
    SUPPORTED_ROBOTS = ['panda_wristcam', 'panda', 'fetch']
    agent: Union[PandaWristCam, Panda, Fetch]

    def __init__(self, *args, robot_uids='panda_wristcam', **kwargs):
        if robot_uids == 'panda_wristcam':
            kwargs['sensor_configs'] = _with_wrist_camera_sensor_config(kwargs.get('sensor_configs'))
        super().__init__(*args, robot_uids=robot_uids, **kwargs)
