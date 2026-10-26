"""EgoNav data-collection tasks built directly on the stable CADRL scenes."""

from isaaclab.utils import configclass

from socnav_sim.tasks.socnav.cadrl.cadrl_env_cfg import CadrlScene12EnvCfg, CadrlScene35EnvCfg


@configclass
class EgoNavScene12EnvCfg(CadrlScene12EnvCfg):
    """Scene 12 with its camera supplied by the recording script."""

    pass


@configclass
class EgoNavScene35EnvCfg(CadrlScene35EnvCfg):
    """Scene 35 with its camera supplied by the recording script."""

    pass
