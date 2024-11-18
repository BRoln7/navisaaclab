# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Humanoid locomotion environment (similar to OpenAI Gym Humanoid-v2).
"""

import gymnasium as gym

from . import agents, pacer_env_cfg

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Pacer",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": pacer_env_cfg.PacerEnvCfg,
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
)
