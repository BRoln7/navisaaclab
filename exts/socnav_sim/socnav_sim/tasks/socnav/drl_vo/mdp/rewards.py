"""Use the same goal and pedestrian-collision rewards as CADRL."""

from socnav_sim.tasks.socnav.cadrl.mdp.rewards import (
    compute_collision_reward,
    compute_goal_reward,
)

__all__ = ["compute_goal_reward", "compute_collision_reward"]
