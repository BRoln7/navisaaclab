from omni.isaac.lab.envs.mdp.actions import JointActionCfg, JointAction, joint_actions, actions_cfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.managers.action_manager import ActionTerm, ActionTermCfg
from omni.isaac.lab.envs import ManagerBasedEnv
from socnav_sim.tasks.amp.smpl_humanoid.mdp.observations import joint_gym_to_sim


class JointPositionPacerAction(JointAction):
    """Joint action term that applies the processed actions to the articulation's joints as position commands."""

    cfg: actions_cfg.JointPositionActionCfg
    """The configuration of the action term."""

    def __init__(self, cfg: actions_cfg.JointPositionActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        if cfg.use_default_offset:
            self._offset = self._asset.data.default_joint_pos[:, self._joint_ids].clone()

    def process_actions(self, actions):
        self._raw_actions[:] = actions
        self._processed_actions = self._raw_actions * self._scale + self._offset
        self._processed_actions[:, joint_gym_to_sim] = self._processed_actions.clone()

    def apply_actions(self):
         self._asset.set_joint_position_target(self.processed_actions, joint_ids=self._joint_ids)


@configclass
class JointPositionActionPacerCfg(JointActionCfg):
    """Configuration for the joint position action term.

    See :class:`JointPositionAction` for more details.
    """

    class_type: type[ActionTerm] = JointPositionPacerAction

    use_default_offset: bool = True
    """Whether to use default joint positions configured in the articulation asset as offset.
    Defaults to True.

    If True, this flag results in overwriting the values of :attr:`offset` to the default joint positions
    from the articulation asset.
    """
