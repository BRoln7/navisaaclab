from omni.isaac.lab.envs.mdp.actions import JointActionCfg, JointAction, joint_actions, actions_cfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.managers.action_manager import ActionTerm, ActionTermCfg
from omni.isaac.lab.envs import ManagerBasedEnv
from socnav_sim.tasks.amp.amp_utils.utils.obs_utils import dof_gym_to_sim
import socnav_sim.assets.nova_carter as carter

import torch
from torch import nn
import numpy as np
from omni.isaac.lab.assets import Articulation
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.markers import VisualizationMarkersCfg, VisualizationMarkers
import omni.isaac.lab.sim as sim_utils
from socnav_sim.tasks.amp.amp_utils.utils.obs_utils import (
    compute_humanoid_observations_smpl_max,
    rot_wxyz_to_xyzw,
    root_states_wxyz_to_xyzw,
    compute_location_observations,
    get_height_maps_jit,
    body_sim_to_gym,
)
from socnav_sim.tasks.amp.amp_utils.utils.running_mean_std import RunningMeanStd
import socnav_sim.tasks.amp.amp_utils.utils.traj_generator as traj_generator
import socnav_sim.tasks.socnav.nav_utils.reset_base as reset
import socnav_sim.tasks.socnav.nav_utils.event_base as event
import os
import yaml

def config_init():
    motion_file = os.path.join(os.getcwd(), 'data', 'cfg', 'task.yaml')
    with open(motion_file, 'r') as file:
        config = yaml.safe_load(file)
    if config['env']['task'] != 'Isaac-Pacer': # type: ignore
        global actions_path, num_envs, num_robots, traj_speed, trajectory_marker, trajectoryog_marker
        trajectory_cfg = VisualizationMarkersCfg(
            prim_path="/World/Visuals/TrajectoryMarkers",
            markers={"waypoints": sim_utils.SphereCfg(
                radius=0.05,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 1.0)))})
        trajectory_marker = VisualizationMarkers(trajectory_cfg)
        actions_path = os.path.join(os.getcwd(), 'data', 'checkpoints', 'actions_policy.pth')
        num_envs = config['env']['num_envs'] # type: ignore
        num_robots = config['env']['num_robots'] # type: ignore
        traj_speed = config['env']['traj_speed'] # type: ignore
config_init()

traj_sample_index = torch.tensor([], device='cuda:0')

class AnimationNet(nn.Module):
    def __init__(self):
        super(AnimationNet, self).__init__()
        
        self.running_mean_std = RunningMeanStd(633)
        self.disc_mlp = nn.Sequential(
            nn.Linear(633, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 69))
    
    def forward(self, obs):
        norm_obs = self.running_mean_std(obs)
        output = self.disc_mlp(norm_obs)
        return output

class NavWithPedAction(JointAction):
    cfg: actions_cfg.JointPositionActionCfg

    def __init__(self, cfg: actions_cfg.JointPositionActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        
        if cfg.use_default_offset:
            self._offset = self._asset.data.default_joint_pos[:, self._joint_ids].clone()
 
        ############### peds ################
        self.animation_net = AnimationNet()
        actions_policy = torch.load(actions_path)
        self.animation_net.load_state_dict(actions_policy)
        self.animation_net.eval()
        self.animation_net.to("cuda:0")
        self._speed_min = 0.0
        self._speed_max = 3.0
        self._accel_max = 2.0
        self._sharp_turn_prob = 0.02
        self._num_traj_samples = 10
        self._traj_sample_timestep = 0.5
        self._traj_gen_init = False
        self._traj_gen = None
        self.progress_buf = None
        self._has_upright_start = True
        self.num_height_points = 1024
        self._has_shape_obs_disc = False
        self.height_sample_width = 1.0
        self.num_samples_per_axis = 16
        self.height_sample_init_flag = False
        self.ped_actions = None
        self.robot_actions = None
        self.ped_scale = 3.1416
        self.ped_offset = 0.0

        self.traj_speed = None

    def process_actions(self, actions):
        ############### peds ################
        humanoid_obs = self.compute_humanoid_obs(self._env)
        task_obs = self.compute_task_obs(self._env)
        terrain_obs = self.compute_height_obs(self._env)
        obs = torch.cat([humanoid_obs, task_obs, terrain_obs], dim=-1)

        with torch.no_grad():
            ped_actions = self.animation_net(obs)
        self.ped_actions = ped_actions * self.ped_scale + self.ped_offset
        self.ped_actions = self.ped_actions[:, dof_gym_to_sim]
        self.freeze_hands_toes(self._env.scene['ped'])

        ############### robots ################
        vx_min = 0
        vx_max = 0.5
        vz_min = -2
        vz_max = 2

        # don't contrl abandon robots
        actions[num_robots:] = 0
        
        self._processed_actions[..., 0] = (actions[..., 0] + 1) * (vx_max - vx_min) / 2 + vx_min # lin_vel
        self._processed_actions[..., 1] = (actions[..., 1] + 1) * (vz_max - vz_min) / 2 + vz_min # ang_vel
        self.robot_actions = carter.two_wheel_differential(self._processed_actions, 
                                                           carter.wheel_radius,
                                                           carter.width,
                                                           self.num_envs)
        
    def apply_actions(self):
         ############### peds ################
         ped_asset: Articulation = self._env.scene['ped']
         ped_asset.set_joint_position_target(self.ped_actions) # type: ignore

         ############### robots ################
         self._asset.set_joint_velocity_target(self.robot_actions) # type: ignore


    def freeze_hands_toes(self, ped_asset: Articulation):
        freeze_joint_name = ['L_Hand_x', 'L_Hand_y', 'L_Hand_z',
                            'R_Hand_x', 'R_Hand_y', 'R_Hand_z',
                            'L_Toe_x', 'L_Toe_y', 'L_Toe_z',
                            'R_Toe_x', 'R_Toe_y', 'R_Toe_z']
        freeze_joint_idx = ped_asset.find_joints(freeze_joint_name)
        self.ped_actions[:, freeze_joint_idx[0]] = 0 # type: ignore

        ############## compute ped's observations #####################
    def compute_humanoid_obs(self, 
                             env: ManagerBasedEnv, 
                             asset_cfg: SceneEntityCfg = SceneEntityCfg("ped")
                        ) -> torch.Tensor:
        asset: Articulation = env.scene[asset_cfg.name]

        upright_config = True
        local_root_obs_config = True
        local_height_obs_config = False

        body_pos = asset.data.body_pos_w.clone()[:, body_sim_to_gym]
        body_rot = rot_wxyz_to_xyzw(asset.data.body_quat_w.clone())[:, body_sim_to_gym]
        body_vel = asset.data.body_lin_vel_w.clone()[:, body_sim_to_gym]
        body_ang_vel = asset.data.body_ang_vel_w.clone()[:, body_sim_to_gym]

        obs = compute_humanoid_observations_smpl_max(body_pos, body_rot, body_vel, body_ang_vel, 
                                                    local_root_obs_config, local_height_obs_config, 
                                                    upright_config)

        return obs
        
    def _build_traj_generator(self, 
                              env: ManagerBasedEnv, 
                              asset_cfg: SceneEntityCfg = SceneEntityCfg("ped")):
        asset: Articulation = env.scene[asset_cfg.name]
        
        episode_dur = env.max_episode_length * env.physics_dt * traj_speed # type: ignore
        num_verts = 101
        dtheta_max = 2.0
        self._traj_gen = traj_generator.TrajGenerator(env.num_envs, episode_dur, num_verts,
                                                      env.device, dtheta_max,
                                                      self._speed_min, self._speed_max,
                                                      self._accel_max, self._sharp_turn_prob)

        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
        root_pos = asset.data.root_pos_w[:, 0:3]
        self._traj_gen.reset(env_ids, root_pos)

        print(f"INFO: traj_gen init")

        return

    def _fetch_traj_samples(self,
                            env: ManagerBasedEnv):
        # 5 seconds with 0.5 second intervals, 10 samples.
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long) # type: ignore

        timestep_beg = self.progress_buf[env_ids] * env.physics_dt * traj_speed # type: ignore
        
        timesteps = torch.arange(self._num_traj_samples, device=env.device, dtype=torch.float)
        timesteps = timesteps * self._traj_sample_timestep
        traj_timesteps = timestep_beg.unsqueeze(-1) + timesteps

        env_ids_tiled = torch.broadcast_to(env_ids.unsqueeze(-1), traj_timesteps.shape)

        traj_samples_flat = self._traj_gen.calc_pos(env_ids_tiled.flatten(), traj_timesteps.flatten()) # type: ignore
        traj_samples = torch.reshape(traj_samples_flat, shape=(env_ids.shape[0], self._num_traj_samples, traj_samples_flat.shape[-1]))

        return traj_samples

    def traj_extension(self, env: ManagerBasedEnv):
        start_index = (self.progress_buf * self.traj_speed).unsqueeze(-1).repeat(1, 10).to(torch.int) # type: ignore
        samples = torch.arange(10, device='cuda:0').unsqueeze(0).repeat(env.num_envs, 1) * \
            torch.ones(env.num_envs, 10, device='cuda:0', dtype=torch.int) * 15 # type: ignore
        final_index = start_index + samples
        assert final_index.max() < event.trajs_p.shape[1], "traj run out!"
        traj_interval = event.trajs_p[torch.arange(env.num_envs, device='cuda:0')[:, None], final_index]
        traj = torch.cat([traj_interval, torch.zeros(env.num_envs, 10, 1, device='cuda:0')], dim=-1)

        return traj, final_index

    def compute_task_obs(self, 
                         env: ManagerBasedEnv, 
                         asset_cfg: SceneEntityCfg = SceneEntityCfg("ped")
                         ) -> torch.Tensor:
        asset: Articulation = env.scene[asset_cfg.name]

        if self._traj_gen_init == False:
            self._traj_gen_init = True
            # self._build_traj_generator(env, asset_cfg)
            self.progress_buf = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
            self.traj_speed = torch.randint(low=1, high=5, size=(env.num_envs,), device='cuda:0') * 0.40

        # for traj_reset
        # root_pos = asset.data.root_pos_w[:, 0:3]
        # if reset.ped_fall_reset is True:
        #     root_pos_reset = root_pos[reset.reset_env_ids, :]
        #     self._traj_gen.reset(reset.reset_env_ids, root_pos_reset) # type: ignore
        #     self.progress_buf[reset.reset_env_ids] = 0 # type: ignore
        #     reset.ped_fall_reset = False

        # traj_ends_env_ids = (self.progress_buf == 600).nonzero(as_tuple=True)[0]
        # root_pos_traj_reset = root_pos[traj_ends_env_ids, ...]
        # self._traj_gen.reset(traj_ends_env_ids, root_pos_traj_reset) # type: ignore
        # self.progress_buf[traj_ends_env_ids] = 0 # type: ignore

        # for traj_reset        
        if reset.ped_reset_flag is True:
            if reset.fall_ids.numel()!=0:
                self.progress_buf[reset.fall_ids] = 0 # type: ignore
            if reset.reach_ids.numel()!=0:
                self.progress_buf[reset.reach_ids] = 0 # type: ignore
            reset.ped_reset_flag = False

        # compute traj_obs
        global traj_sample_index
        traj_samples, traj_sample_index = self.traj_extension(env)
        # traj_samples = self._fetch_traj_samples(env)
        self.progress_buf += 1 # type: ignore
        root_states = root_states_wxyz_to_xyzw(asset.data.root_state_w)
        obs = compute_location_observations(root_states, traj_samples, self._has_upright_start)

        trajectory_marker.visualize(translations=traj_samples.reshape(-1, 3))
        # trajectoryog_marker.visualize(translations=traj_samples.reshape(-1, 3))

        return obs
    
    def compute_height_obs(self,
                           env: ManagerBasedEnv, 
                           asset_cfg: SceneEntityCfg = SceneEntityCfg("ped")
                        ) -> torch.Tensor:
        if self.height_sample_init_flag is False:
            self.height_points = self.init_height_points()
            self.height_sample_init_flag = True

        asset: Articulation = env.scene[asset_cfg.name]
        root_states = root_states_wxyz_to_xyzw(asset.data.root_state_w.clone())[..., :7]
        base_pos = root_states[..., :3]

        self.height_samples = torch.zeros((2800, 1400), device='cuda:0')

        height_obs = get_height_maps_jit(
            root_states=root_states,
            base_pos=base_pos,
            env_ids=torch.arange(0, env.num_envs, 1).to('cuda:0'),
            num_envs=num_envs,
            height_points=self.height_points,
            height_samples=self.height_samples,
            num_height_points=self.num_height_points,
            terrain_horizontal_scale=0.1,
            w_last=True,
            return_all_dims=False,
        )

        return height_obs
    
    def init_height_points(self):
        """
        Pre-defines the grid for the height-map observation.
        """
        y = torch.tensor(
            np.linspace(
                -self.height_sample_width,
                self.height_sample_width,
                self.num_samples_per_axis,
            ),
            device="cuda:0",
            requires_grad=False,
        )
        x = torch.tensor(
            np.linspace(
                -self.height_sample_width,
                self.height_sample_width,
                self.num_samples_per_axis,
            ),
            device="cuda:0",
            requires_grad=False,
        )
        grid_x, grid_y = torch.meshgrid(x, y)

        self.num_height_points = grid_x.numel()
        points = torch.zeros(
            num_envs,
            self.num_height_points,
            3,
            device='cuda:0',
            requires_grad=False,
        )
        points[:, :, 0] = grid_x.flatten()
        points[:, :, 1] = grid_y.flatten()
        return points
    


@configclass
class NavWithPedActionCfg(JointActionCfg):
    class_type: type[ActionTerm] = NavWithPedAction
    use_default_offset: bool = True
