"""Register robot-egocentric social-navigation data collection tasks."""

import gymnasium as gym


for scene_id in (12, 35):
    gym.register(
        id=f"Isaac-EgoNav-Scene{scene_id}",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.egonav_env_cfg:EgoNavScene{scene_id}EnvCfg",
            "sb3_cfg_entry_point": "socnav_sim.tasks.socnav.cadrl.agents:sb3_ppo_cfg.yaml",
        },
    )
