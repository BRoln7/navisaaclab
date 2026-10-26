"""Register the public CADRL navigation examples."""

import gymnasium as gym


for scene_id in (12, 27, 34, 35):
    gym.register(
        id=f"Isaac-Cadrl-Scene{scene_id}",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.cadrl_env_cfg:CadrlScene{scene_id}EnvCfg",
            "sb3_cfg_entry_point": f"{__name__}.agents:sb3_ppo_cfg.yaml",
        },
    )
