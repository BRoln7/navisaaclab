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
        self.freeze_hand_toe()

    def apply_actions(self):
         self._asset.set_joint_position_target(self.processed_actions, joint_ids=self._joint_ids)

    def freeze_hand_toe(self):
        freeze_joint_name = ['L_Hand_x', 'L_Hand_y', 'L_Hand_z',
                            'R_Hand_x', 'R_Hand_y', 'R_Hand_z',
                            'L_Toe_x', 'L_Toe_y', 'L_Toe_z',
                            'R_Toe_x', 'R_Toe_y', 'R_Toe_z']
        freeze_joint_idx = self._asset.find_joints(freeze_joint_name)
        self._processed_actions[:, freeze_joint_idx[0]] = 0

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
