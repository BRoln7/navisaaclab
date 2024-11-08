from omni.isaac.lab.envs.mdp.actions import JointActionCfg, JointAction, joint_actions, actions_cfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.managers.action_manager import ActionTerm, ActionTermCfg
from omni.isaac.lab.envs import ManagerBasedEnv
import torch
from socnav_sim.assets.jackal import jackal_lenght, jackal_width, jackal_wheel_radius
from socnav_sim.assets.turtlebot3 import turtlebot_width, turtlebot_wheel_radius

class JointVelocityJackalAction(JointAction):
    """Joint action term that applies the processed actions to the articulation's joints as position commands."""

    cfg: actions_cfg.JointVelocityActionCfg
    """The configuration of the action term."""

    def __init__(self, cfg: actions_cfg.JointVelocityActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        if cfg.use_default_offset:
            self._offset = self._asset.data.default_joint_pos[:, self._joint_ids].clone()

    def process_actions(self, actions):
        self._raw_actions[:] = actions
        self._processed_actions = self.four_wheel_differential(actions, jackal_wheel_radius, jackal_lenght, jackal_width)

    def apply_actions(self):
        self._asset.set_joint_velocity_target(self.processed_actions, joint_ids=self._joint_ids)

    def four_wheel_differential(self, twist, wheel_r, length, width):
        """
        twist: [num_envs, 2-dimension twist]
        wheel_r: radius of wheel
        length: distance of front and rear wheel
        width: distance of left and right wheel 
        """
        velocity_tensor = torch.zeros(self.num_envs, 4).to("cuda:0")
        velocity_tensor[:, 0] = twist[:, 0] - twist[:, 1] * length / 2 + twist[:, 1] * width / 2
        velocity_tensor[:, 1] = twist[:, 0] - twist[:, 1] * length / 2 - twist[:, 1] * width / 2
        velocity_tensor[:, 2] = twist[:, 0] + twist[:, 1] * length / 2 + twist[:, 1] * width / 2
        velocity_tensor[:, 3] = twist[:, 0] + twist[:, 1] * length / 2 - twist[:, 1] * width / 2
        velocity_tensor = velocity_tensor.clone() / wheel_r

        return velocity_tensor


@configclass
class JointVelocityActionJackalCfg(JointActionCfg):
    """Configuration for the joint position action term.

    See :class:`JointPositionAction` for more details.
    """

    class_type: type[ActionTerm] = JointVelocityJackalAction

    use_default_offset: bool = True
    """Whether to use default joint positions configured in the articulation asset as offset.
    Defaults to True.

    If True, this flag results in overwriting the values of :attr:`offset` to the default joint positions
    from the articulation asset.
    """


class JointVelocityTurtlebotAction(JointAction):
    """Joint action term that applies the processed actions to the articulation's joints as position commands."""

    cfg: actions_cfg.JointVelocityActionCfg
    """The configuration of the action term."""

    def __init__(self, cfg: actions_cfg.JointVelocityActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        if cfg.use_default_offset:
            self._offset = self._asset.data.default_joint_pos[:, self._joint_ids].clone()

    def process_actions(self, actions):
        self._raw_actions[:] = actions
        self._processed_actions = self.two_wheel_differential(actions, turtlebot_wheel_radius, turtlebot_width)

    def apply_actions(self):
        self._asset.set_joint_velocity_target(self.processed_actions, joint_ids=self._joint_ids)

    def two_wheel_differential(self, twist, wheel_r, width):
        """
        twist: [num_envs, 2-dimension twist]
        wheel_r: radius of wheel
        width: distance of left and right wheel 
        """
        velocity_tensor = torch.zeros(self.num_envs, 2).to("cuda:0")
        velocity_tensor[:, 0] = twist[:, 0] + twist[:, 1] * width / 2
        velocity_tensor[:, 1] = twist[:, 0] - twist[:, 1] * width / 2
        velocity_tensor = velocity_tensor.clone() / wheel_r

        return velocity_tensor


@configclass
class JointVelocityActionTurtlebotCfg(JointActionCfg):
    """Configuration for the joint position action term.

    See :class:`JointPositionAction` for more details.
    """

    class_type: type[ActionTerm] = JointVelocityTurtlebotAction

    use_default_offset: bool = True
    """Whether to use default joint positions configured in the articulation asset as offset.
    Defaults to True.

    If True, this flag results in overwriting the values of :attr:`offset` to the default joint positions
    from the articulation asset.
    """