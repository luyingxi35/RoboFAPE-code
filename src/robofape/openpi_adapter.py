# Copyright 2025 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
# Modified: standalone two-camera inference configuration; no training runner.
"""Pi0.5 two-camera transform configuration for exported OpenPI checkpoints."""
from __future__ import annotations
import dataclasses
import numpy as np


def create_policy(checkpoint, config):
    from openpi import transforms
    from openpi.models.pi0_config import Pi0Config
    from openpi.policies.policy_config import create_trained_policy
    from openpi.training.config import AssetsConfig, DataConfig, DataConfigFactory, ModelTransformFactory, TrainConfig

    @dataclasses.dataclass(frozen=True)
    class Inputs(transforms.DataTransformFn):
        def __call__(self, data):
            main=np.asarray(data["observation/image"],dtype=np.uint8)
            wrist=np.asarray(data["observation/wrist_image"],dtype=np.uint8)
            return {"state":data["observation/state"], "prompt":data["prompt"],
                    "image":{"base_0_rgb":main,"left_wrist_0_rgb":wrist,"right_wrist_0_rgb":np.zeros_like(main)},
                    "image_mask":{"base_0_rgb":np.True_,"left_wrist_0_rgb":np.True_,"right_wrist_0_rgb":np.False_}}

    @dataclasses.dataclass(frozen=True)
    class Outputs(transforms.DataTransformFn):
        def __call__(self, data):
            return {"actions":np.asarray(data["actions"][:,:config["action_dim"]])}

    @dataclasses.dataclass(frozen=True)
    class InferenceData(DataConfigFactory):
        def create(self, assets_dirs, model_config):
            return dataclasses.replace(self.create_base_config(assets_dirs,model_config),
                data_transforms=transforms.Group(inputs=[Inputs()],outputs=[Outputs()]),
                model_transforms=ModelTransformFactory()(model_config))

    if "asset_id" not in config or "discrete_state_input" not in config:
        raise ValueError("Pi0.5 config requires the checkpoint's asset_id and discrete_state_input")
    train_config=TrainConfig(name="robofape_pi05_inference",model=Pi0Config(pi05=True,
        action_horizon=config.get("action_horizon",10),discrete_state_input=config["discrete_state_input"]),
        data=InferenceData(repo_id=config["asset_id"],base_config=DataConfig(prompt_from_task=True),
                          assets=AssetsConfig(assets_dir=str(checkpoint/"assets"),asset_id=config["asset_id"])))
    return create_trained_policy(train_config,str(checkpoint))
