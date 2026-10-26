"""Register DRL-VO without loading assets or simulation modules."""

import gymnasium as gym


for task, config in (
    ("Isaac-Drlvo", "DrlvoScene12EnvCfg"),
    ("Isaac-Drlvo-Scene12", "DrlvoScene12EnvCfg"),
    ("Isaac-Drlvo-Scene35", "DrlvoScene35EnvCfg"),
):
    gym.register(
        id=task,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.drlvo_env_cfg:{config}",
            "sb3_cfg_entry_point": f"{__name__}.agents:sb3_ppo_cfg.yaml",
        },
    )
