"""Small Isaac Lab backend extension for the official MimicKit SMPL USD/XML pair.

Import only AFTER AppLauncher. Native MimicKit keeps its scene, physics, contact,
body ordering and actuation implementation. This subclass owns no RL logic.
"""

from pathlib import Path
import torch
from engines.isaac_lab_engine import IsaacLabEngine


class SmplIsaacLabEngine(IsaacLabEngine):
    def __init__(self, config, *args, **kwargs):
        self._smpl_usd = str(Path(config["smpl_usd"]).resolve())
        self._smpl_self_collisions = bool(config.get("enable_self_collisions", True))
        super().__init__(config, *args, **kwargs)

    def create_obj(self, env_id, obj_type, asset_file, name, **kwargs):
        kwargs.setdefault("enable_self_collisions", self._smpl_self_collisions)
        return super().create_obj(env_id, obj_type, asset_file, name, **kwargs)

    def _create_simulator(self, sim_timestep, visualize, record_video):
        # The entry point has launched SimulationApp, before upstream imports carb.
        import carb
        import isaaclab.sim as sim_utils
        from isaacsim.core.utils.stage import get_current_stage
        cfg = sim_utils.SimulationCfg(device=self._device, dt=sim_timestep, render_interval=self._sim_steps)
        cfg.physx.bounce_threshold_velocity = .2
        cfg.physx.max_position_iteration_count = 4
        cfg.physx.max_velocity_iteration_count = 0
        cfg.physx.gpu_max_rigid_contact_count = 8*1024*1024
        cfg.physics_material.static_friction = 1.
        cfg.physics_material.dynamic_friction = 1.
        self._sim = sim_utils.SimulationContext(cfg)
        self._stage = get_current_stage()
        carb.settings.get_settings().set_bool("/app/runLoops/main/rateLimitEnabled", False)

    def _parse_usd_path(self, asset_file):
        # Native MimicKit resolves smpl.xml -> the supplied sibling smpl.usd.
        path = super()._parse_usd_path(asset_file)
        if str(Path(path).resolve()) != self._smpl_usd:
            raise ValueError("SMPL physics must use the USD paired with the configured XML")
        return path

    def get_dof_forces(self, obj_id):
        """Return Isaac Lab's clipped implicit-PD torque estimate in common DOF order.

        PhysX does not expose the instantaneous internal drive torque used by
        Isaac Gym's DOF-force sensor.  ``applied_torque`` is Isaac Lab's
        supported reward signal for an implicit actuator; it is recomputed at
        every physics substep by ``write_data_to_sim``.
        """
        obj = self._objs[obj_id]
        torque = obj.data.applied_torque
        return torque[:, self._dof_order_sim2common[obj_id]]

    def _build_body_order(self, obj):
        from .data import SMPL_DFS
        meta = obj.root_physx_view.shared_metatype
        names = list(meta.link_names)
        if set(names) != set(SMPL_DFS) or len(names) != 24:
            raise ValueError("Expected the SMPL USD's 24 bodies and 69 spherical DOFs")
        bodies = [names.index(n) for n in SMPL_DFS]
        dofs = []
        for body in bodies[1:]:
            count = meta.joint_dof_counts[body-1]
            if count != 3:
                raise ValueError("Each SMPL child joint must have three rotational DOFs")
            start = meta.joint_dof_offsets[body-1]
            dofs.extend(range(start, start+count))
        return bodies, [bodies.index(i) for i in range(24)], dofs, [dofs.index(i) for i in range(69)]

    def flush_resets(self, obj_id, env_ids):
        """Commit reset states without stepping time; start PD at the sampled pose."""
        self._update_reset_objs()
        obj = self._objs[obj_id]
        obj.set_joint_position_target(obj.data.joint_pos[env_ids], env_ids=env_ids)
        obj.set_joint_velocity_target(torch.zeros_like(obj.data.joint_vel[env_ids]), env_ids=env_ids)
        self._sim.forward()
        for asset in self._objs:
            # Advance the DATA cache timestamp, not physics/environment time.
            # update(0) can leave the manually populated FK cache valid, hiding
            # a USD joint convention mismatch from validate_reset().
            asset.update(self.get_sim_timestep())
