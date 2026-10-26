"""CPU-only contract tests for the independent MimicKit integration."""

import json
import hashlib
import math
import pickle
from pathlib import Path
import sys
from types import SimpleNamespace

import gymnasium as gym
import joblib
import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "exts/socnav_sim"))
from socnav_sim.mimickit_amp.bootstrap import enable_mimickit
enable_mimickit()
from socnav_sim.mimickit_amp.agent import SymmetricAMPAgent
from socnav_sim.mimickit_amp.buffers import StagedExperienceBuffer
from socnav_sim.mimickit_amp.config import latest_checkpoint, load, resolve_agent_runtime, resolve_paths, run_configurations
from socnav_sim.mimickit_amp.data import SMPL_DFS, convert
from socnav_sim.mimickit_amp.data import sha256
from socnav_sim.mimickit_amp.sampling import build_running_sampling, classify_clip, speed_statistics
from socnav_sim.mimickit_amp.env import SmplTrackingEnv, mechanical_power, sample_history_times, trajectory_failure
from socnav_sim.mimickit_amp.symmetry import SmplReflection
from socnav_sim.mimickit_amp.trajectory import Trajectory
from anim.mjcf_char_model import MJCFCharModel
from anim.motion_lib import MotionLib
from envs.char_env import compute_char_obs
from learning.base_agent import AgentMode
from learning.amp_agent import AMPAgent
from learning.experience_buffer import ExperienceBuffer
from util import mp_util, torch_util

KEYS = ["L_Toe", "R_Toe", "Head", "L_Hand", "R_Hand"]


@pytest.mark.parametrize("num_envs", [16, 64, 4096, 10003, 12288, 16384])
def test_shared_agent_adapts_batches_and_replay_after_environment_change(num_envs):
    original = load(ROOT / "data/cfg/mimickit_amp/agent.yaml")
    # Also simulate loading the resolved config saved by a 12,288-env run.
    saved = resolve_agent_runtime(original, 12288)
    for source in (original, saved):
        resolved = resolve_agent_runtime(source, num_envs)
        assert resolved["disc_buffer_size"] == 200000
        for name, per_env, limit in (("actor", 4, 16384), ("critic", 2, 8192), ("disc", 2, 4096)):
            assert math.ceil(resolved[f"{name}_batch_size"]*num_envs) == per_env*num_envs
            assert resolved["optimizer_microbatch_size"][name] == limit
            steps = math.ceil(32*num_envs/(per_env*num_envs)) * resolved[f"{name}_epochs"]
            assert steps == (40 if name == "actor" else 32)
        assert resolved["symmetry_weight"] == original["symmetry_weight"]
        assert resolved["task_reward_weight"] == original["task_reward_weight"]
        assert resolved["disc_reward_weight"] == original["disc_reward_weight"]
    assert original["disc_buffer_size"] == 200000


def test_configs_inherit_native_parameters_with_tuned_symmetry_override():
    native_agent = load("third_party/MimicKit/data/agents/amp_smpl_agent.yaml")
    agent = load("data/cfg/mimickit_amp/agent.yaml")
    for key, value in native_agent.items():
        if key != "symmetry_weight":
            assert agent[key] == value, key
    native_env = load("third_party/MimicKit/data/envs/amp_smpl_env.yaml")
    env = load("data/cfg/mimickit_amp/env.yaml")
    for key, value in native_env.items():
        if key not in ("char_file", "motion_file"):
            assert env[key] == value, key
    assert env["motion_file"] == "data/mimickit/amass-pacer-v1/native/motions.yaml"
    assert env["tracking"]["enabled"] is False
    assert agent["symmetry_weight"] == 2.
    assert agent["symmetry_scale"] == 50.
    assert env["tracking"] == {
        "enabled": False,
        "reward_weight": .5,
        "style_reward_weight": .5,
        "num_vertices": 101,
        "heading_change_max": 2.,
        "observation_samples": 10,
        "observation_timestep": .5,
        "speed_min": 0.,
        "speed_max": 3.,
        "acceleration_max": 2.,
        "sharp_turn_probability": .02,
        "position_scale": 2.,
        "max_distance": 4.,
            "power_coefficient": 0.,
    }


def test_legacy_capped_batches_migrate_to_native_logical_batches():
    cfg = load("data/cfg/mimickit_amp/agent.yaml")
    cfg.pop("optimizer_microbatch_size")
    cfg.update(actor_batch_size=1., critic_batch_size=.5, disc_batch_size=.5,
               disc_buffer_size=524288, disc_buffer_size_min=200000,
               batch_size_scaling={name: {"per_env": ratio, "max_samples": limit}
                                   for name, ratio, limit in (("actor", 4, 16384), ("critic", 2, 8192), ("disc", 2, 8192))})
    migrated = resolve_agent_runtime(cfg, 16384)
    assert migrated["disc_buffer_size"] == 200000
    assert migrated["actor_batch_size"] == 4
    assert migrated["critic_batch_size"] == migrated["disc_batch_size"] == 2
    assert migrated["optimizer_microbatch_size"] == dict(actor=16384, critic=8192, disc=8192)
    assert "batch_size_scaling" not in migrated and "disc_buffer_size_min" not in migrated


@pytest.mark.parametrize("invalid", [0, -1, 1.5, True])
def test_optimizer_microbatch_size_must_be_a_positive_integer(invalid):
    cfg = load("data/cfg/mimickit_amp/agent.yaml")
    cfg["optimizer_microbatch_size"]["disc"] = invalid
    with pytest.raises(ValueError, match="positive integer"):
        resolve_agent_runtime(cfg, 16384)


@pytest.mark.parametrize("name,speed,group", [
    ("0-KIT_317_walking_run08_poses", 3., "walking_label"),
    ("0-ACCAD_Male1Running_c3d_Run C24 - quick side step left_poses", 3., "lateral_step"),
    ("0-ACCAD_Male2Walking_c3d_B18 - Walk to leap to walk_poses", 3., "walk_hop_leap"),
    ("0-ACCAD_Female1Running_c3d_C5 - walk to run_poses", 3., "walk_run_transition"),
    ("0-ACCAD_Female1Running_c3d_C4 - Run to walk1_poses", 3., "walk_run_transition"),
    ("0-KIT_424_run02_poses", 3., "running_candidate"),
    ("0-KIT_424_run02_poses", .5, "other_or_uncertain"),
    ("0-HumanEva_S2_Jog_1_poses", 1., "jogging_label"),
])
def test_sampling_categories_do_not_confuse_recording_names_and_gaits(name, speed, group):
    assert classify_clip(name, speed) == group


def test_sampling_speed_uses_path_length_and_legal_ten_frame_history():
    frames = np.zeros((31, 75))
    frames[:, 0] = np.r_[np.arange(16)*.1, np.arange(14, -1, -1)*.1]
    speed, windows = speed_statistics(frames, 30., .3)
    np.testing.assert_allclose(speed, 3.)
    np.testing.assert_allclose(windows, 3.)
    assert len(windows) == 22  # n_frames - nine history intervals.
    assert frames[-1, 0] == frames[0, 0]  # Net displacement would wrongly say zero.


def test_running_reweighting_reuses_unique_clips_and_native_probabilities(tmp_path):
    import yaml
    entries, motions = [], []
    for i, (name, speed, weight) in enumerate((("0-KIT_1_walking_run01_poses", 1., 2.),
                                               ("0-KIT_1_run01_poses", 3., 1.),
                                               ("0-HumanEva_S2_Jog_1_poses", 1.2, .5))):
        path = tmp_path / f"clip_{i:04d}.pkl"
        frames = np.zeros((61, 75), dtype=np.float32)
        frames[:, 0] = np.arange(61)*speed/30
        frames[:, 2] = 1.
        with path.open("wb") as stream:
            pickle.dump(dict(loop_mode=0, fps=30., frames=frames.tolist()), stream)
        entries.append(dict(file=path.name, name=name, sha256=sha256(path)))
        motions.append(dict(file=str(path), weight=weight))
    (tmp_path / "manifest.json").write_text(json.dumps(dict(clips=entries, history_steps=10, control_dt=1/30)))
    source, output = tmp_path / "motions.yaml", tmp_path / "motions-run6.yaml"
    source.write_text(yaml.safe_dump(dict(motions=motions)))
    original_hash = sha256(source)
    report = build_running_sampling(source, output, running_weight=6.)
    assert report["boosted_clips"] == ["clip_0001.pkl", "clip_0002.pkl"]
    assert sha256(source) == original_hash
    assert build_running_sampling(source, output, running_weight=6.) == report
    with pytest.raises(FileExistsError, match="differs"):
        build_running_sampling(source, output, running_weight=2.)
    model = MJCFCharModel("cpu"); model.load(str(ROOT / "data/cfg/smpl_skeletontree.xml"))
    library = MotionLib(str(output), model, "cpu")
    assert library.get_num_motions() == 3  # No duplicate pose storage.
    torch.testing.assert_close(library.get_motion_weights(), torch.tensor([2., 6., 3.])/11)
    assert report["new_history_sampling_share"]["2.0_to_2.5"] == 0.
    assert report["new_history_sampling_share"]["at_least_2.5"] == pytest.approx(6/11)


@pytest.fixture(autouse=True)
def cpu_only():
    torch.set_num_threads(1)
    assert not torch.cuda.is_initialized()
    if not mp_util.get_num_procs():
        mp_util.init(0, 1, "cpu", 29500)
    yield
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("global_obs", [False, True])
def test_reflection_matches_native_observation_geometry(global_obs):
    torch.manual_seed(42)
    reflection = SmplReflection(SMPL_DFS, KEYS)
    root = torch.randn(5, 3)
    root_q = torch_util.exp_map_to_quat(torch.randn(5, 3))
    vel, omega = torch.randn(5, 3), torch.randn(5, 3)
    joints = torch_util.exp_map_to_quat(torch.randn(5, 23, 3))
    dof_vel = torch.randn(5, 69)
    key_pos = torch.randn(5, 5, 3)
    obs = compute_char_obs(root, root_q, vel, omega, joints, dof_vel, key_pos, global_obs, True)
    mirrored_q = lambda q: q * torch.tensor([-1., 1., -1., 1.])
    other = compute_char_obs(root*reflection.polar, mirrored_q(root_q), vel*reflection.polar, omega*reflection.axial,
                             mirrored_q(joints[:, reflection.joints]), reflection.action(dof_vel),
                             key_pos[:, reflection.keys]*reflection.polar, global_obs, True)
    future = torch.randn(5, 20)
    actual = reflection.observation(torch.cat((obs, future), -1))
    expected = torch.cat((other, (future.reshape(5, 10, 2)*torch.tensor([1., -1.])).flatten(1)), -1)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=1e-5)
    torch.testing.assert_close(reflection.observation(actual), torch.cat((obs, future), -1))
    actions = torch.randn(5, 69)
    torch.testing.assert_close(reflection.action(reflection.action(actions)), actions)


def test_history_sampling_never_clamps_the_past():
    library = SimpleNamespace(sample_motions=lambda n: torch.zeros(n, dtype=torch.long),
                              get_motion_length=lambda ids: torch.ones_like(ids, dtype=torch.float))
    ids, times = sample_history_times(library, 1000, .3)
    assert times.min() >= .3 and times.max() <= 1.
    _, first = sample_history_times(library, 2, .3, random=False)
    torch.testing.assert_close(first, torch.full((2,), .3))
    library.get_motion_length = lambda ids: torch.full(ids.shape, .2)
    with pytest.raises(ValueError, match="history"):
        sample_history_times(library, 3, .3)


def test_paths_are_timed_and_partial_resets_are_isolated():
    path = Trajectory(2, 10, "cpu", speed_min=1., speed_max=1.,
                      acceleration_max=0., heading_change_max=0., sharp_turn_probability=0.)
    ids = torch.arange(2)
    q = torch.tensor([[0., 0., 0., 1.]]).repeat(2, 1)
    torch.manual_seed(20)
    path.reset(ids, torch.zeros(2, 3), q)
    assert path.count == 101 and path.segment_dt == pytest.approx(.1)
    assert path.duration == pytest.approx(10.1)  # PACER's 101 * 0.1 convention.
    segment_length = torch.linalg.vector_norm(torch.diff(path.paths[..., :2], dim=1), dim=-1)
    torch.testing.assert_close(segment_length, torch.full_like(segment_length, .1))
    # Position interpolation follows PACER's duration rather than direct t / segment_dt.
    expected_distance = torch.tensor([1., 2.]) / path.duration * 10.
    torch.testing.assert_close(
        torch.linalg.vector_norm(path.position(ids, torch.tensor([1., 2.]))[:, :2], dim=-1),
        expected_distance)
    untouched = path.paths[1].clone()
    path.reset(ids[:1], torch.ones(1, 3), q[:1])
    torch.testing.assert_close(path.paths[1], untouched)
    a = path.observation(ids, torch.zeros(2, 3), q, torch.zeros(2))
    b = path.observation(ids, torch.zeros(2, 3), q, torch.zeros(2))
    torch.testing.assert_close(a, b)
    samples_a = path.samples(ids, torch.zeros(2))
    samples_b = path.samples(ids, torch.full((2,), .1))
    assert samples_a.shape == (2, 10, 3)
    assert not torch.equal(samples_a, samples_b)
    torch.testing.assert_close(samples_a[..., :2].flatten(1), a)


def test_trajectory_sampling_matches_pacer_draw_order_and_geometry():
    """Lock the port to nv-tlabs/pacer TrajGenerator.reset at baeb624."""
    n, vertices, episode = 3, 8, 2.1
    dt = episode / (vertices - 1)
    positions = torch.tensor([[1., 2., .9], [-1., .5, 1.], [0., 0., .8]])
    ids = torch.arange(n)
    torch.manual_seed(104)
    trajectory = Trajectory(n, episode, "cpu", num_vertices=vertices,
                            heading_change_max=2., speed_min=0., speed_max=3.,
                            acceleration_max=2., sharp_turn_probability=.5)
    trajectory.reset(ids, positions)

    torch.manual_seed(104)
    dtheta = (2 * torch.rand(n, vertices - 1) - 1) * 2. * dt
    sharp = math.pi * (2 * torch.rand(n, vertices - 1) - 1)
    mask = torch.bernoulli(torch.full_like(dtheta, .5)).bool()
    assert mask.any()  # Exercise PACER's sharp-heading replacement path.
    dtheta[mask] = sharp[mask]
    dtheta[:, 0] = math.pi * (2 * torch.rand(n) - 1)
    dspeed = (2 * torch.rand(n, vertices - 1) - 1) * 2. * dt
    dspeed[:, 0] = 3. * torch.rand(n)
    speed = torch.zeros_like(dspeed)
    speed[:, 0] = dspeed[:, 0]
    for i in range(1, vertices - 1):
        speed[:, i] = (speed[:, i - 1] + dspeed[:, i]).clamp(0., 3.)
    heading = dtheta.cumsum(-1)
    displacement = torch.stack((heading.cos(), -heading.sin(), torch.zeros_like(heading)), -1)
    displacement *= (speed * dt).unsqueeze(-1)
    displacement[:, 0, :2] += positions[:, :2]
    expected = torch.zeros_like(trajectory.paths)
    expected[:, 0, :2] = positions[:, :2]
    expected[:, 1:] = displacement.cumsum(1)
    torch.testing.assert_close(trajectory.paths, expected)


@pytest.mark.parametrize("skeleton", [
    ROOT / "data/cfg/smpl_skeletontree.xml",
    pytest.param(ROOT / "third_party/MimicKit/data/assets/smpl/smpl.xml",
                 marks=pytest.mark.skipif(not (ROOT / "third_party/MimicKit/data/assets/smpl/smpl.xml").is_file(),
                                          reason="Official model is installed separately")),
], ids=["legacy-source-fixture", "official-smpl"])
def test_native_motion_conversion_preserves_rotations_and_flight_height(tmp_path, skeleton):
    model = MJCFCharModel("cpu")
    model.load(str(skeleton))
    n = 32
    root = torch.zeros(n, 3); root[:, 0] = torch.arange(n)/30; root[:, 2] = 1.+.1*torch.sin(torch.linspace(0, 3.14, n))
    root_q = torch_util.exp_map_to_quat(torch.tensor([[.03, -.05, .1]]).repeat(n, 1))
    local_q = torch_util.exp_map_to_quat(torch.randn(n, 23, 3)*.1)
    positions, global_q = model.forward_kinematics(root, root_q, local_q)
    source = tmp_path / "input.pkl"
    joblib.dump({"synthetic": {"fps": 30, "root_trans_offset": root, "pose_quat_global": global_q.numpy()}}, source)
    report = convert(source, skeleton, tmp_path / "converted")
    lib = MotionLib(str(tmp_path / "converted/motions.yaml"), model, "cpu")
    values = lib.calc_motion_frame(torch.zeros(n, dtype=torch.long), torch.arange(n)/30)
    reconstructed, recon_q = model.forward_kinematics(values[0], values[1], values[4])
    expected = positions.clone(); expected[..., 2] += report["clips"][0]["floor_shift"]
    torch.testing.assert_close(reconstructed, expected, atol=2e-5, rtol=1e-5)
    # Upstream SLERP/FK accumulates float32 quaternion norm drift along the tree.
    cosine = torch.nn.functional.cosine_similarity(recon_q, global_q, dim=-1).abs()
    torch.testing.assert_close(cosine, torch.ones(n, 24), atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(torch.diff(values[0][:, 2]), torch.diff(root[:, 2]), atol=1e-6, rtol=1e-5)
    assert convert(source, skeleton, tmp_path / "converted") == report


class FakeEnv:
    """Tests upstream learner plumbing only; no fake simulator or locomotion claim."""
    def __init__(self):
        self.reflection = SmplReflection(SMPL_DFS, KEYS)
        self.obs = torch.randn(2, 255)*.1
        self.disc = torch.randn(2, 2370)*.1
        self.t = 0

    def get_obs_space(self):
        return gym.spaces.Box(-np.inf, np.inf, (255,), dtype=np.float32)

    def get_action_space(self):
        return gym.spaces.Box(-2., 2., (69,), dtype=np.float32)

    def get_disc_obs_space(self):
        return gym.spaces.Box(-np.inf, np.inf, (2370,), dtype=np.float32)

    def get_num_envs(self): return 2
    def get_reward_succ(self): return 0.
    def get_reward_fail(self): return 0.
    def set_mode(self, mode): pass
    def get_visualize(self): return False
    def reflect_observation(self, obs): return self.reflection.observation(obs)
    def reflect_action(self, action): return self.reflection.action(action)
    def fetch_disc_obs_demo(self, n): return torch.randn(n, 2370)*.2

    def reset(self, env_ids=None):
        ids = torch.arange(2) if env_ids is None else env_ids
        self.obs[ids] = 0.; self.disc[ids] = 0.
        return self.obs, {"disc_obs": self.disc}

    def step(self, actions):
        self.t += 1
        self.obs[:] = torch.randn_like(self.obs)*.1
        self.disc[:] = torch.randn_like(self.disc)*.1
        self.disc[:, 0] = self.t
        done = torch.tensor([1, 3]) if self.t == 3 else torch.zeros(2, dtype=torch.int)
        return self.obs, torch.full((2,), .5), done, {"disc_obs": self.disc}


def small_agent():
    cfg = load("data/cfg/mimickit_amp/agent.yaml")
    for key in ("actor_net", "critic_net", "disc_net"):
        cfg["model"][key] = "fc_2layers_256units"
    cfg.update(steps_per_iter=4, actor_batch_size=1, critic_batch_size=1, disc_batch_size=1,
               actor_epochs=1, critic_epochs=1, disc_epochs=1, disc_buffer_size=16, iters_per_output=1, test_episodes=2)
    return SymmetricAMPAgent(cfg, FakeEnv(), "cpu")


def test_native_amp_rollout_update_and_checkpoint(tmp_path):
    torch.manual_seed(4)
    agent = small_agent()
    agent._data_batch_size = 3
    agent._curr_obs, agent._curr_info = agent._reset_envs()
    agent._init_train()
    before = {key: value.clone() for key, value in agent._model.state_dict().items()}
    info = agent._train_iter()  # the real upstream rollout, GAE, PPO, replay and discriminator updates
    assert all(torch.isfinite(value).all() for value in info.values() if torch.is_tensor(value))
    assert info["sym_loss"] >= 0
    terminal_history = agent._exp_buffer.get_data("disc_obs")[2, :, 0]
    torch.testing.assert_close(terminal_history, torch.tensor([3., 3.]))
    for key in ("_actor_layers.0.weight", "_critic_layers.0.weight", "_disc_layers.0.weight"):
        assert not torch.equal(before[key], agent._model.state_dict()[key])
    assert agent._obs_norm.get_count().item() > 0
    assert agent._disc_obs_norm.get_count().item() > 0
    # Reuse the same host demo allocation across subsequent iterations and
    # exercise native optimization after replay becomes full.
    demo_storage = agent._exp_buffer.get_data_flat("disc_obs_demo").data_ptr()
    for _ in range(2):
        info = agent._train_iter()
        assert all(torch.isfinite(value).all() for value in info.values() if torch.is_tensor(value))
        assert agent._exp_buffer.get_data_flat("disc_obs_demo").data_ptr() == demo_storage
    assert agent._disc_buffer.is_full()
    assert agent._disc_buffer.get_data_flat("disc_obs").device.type == "cpu"
    assert agent._exp_buffer.get_data_flat("disc_obs_demo").device.type == "cpu"
    checkpoint = tmp_path / "model.pt"
    agent.save(str(checkpoint))
    other = small_agent(); other.load(str(checkpoint))
    for key, value in agent.state_dict().items():
        torch.testing.assert_close(value, other.state_dict()[key])
    agent.set_mode(AgentMode.TEST); other.set_mode(AgentMode.TEST)
    obs = torch.randn(2, 255)*.1
    torch.testing.assert_close(agent._decide_action(obs, {})[0], other._decide_action(obs, {})[0])


def test_chunked_demo_normalization_rewards_and_replay_match_upstream():
    actual, reference = small_agent(), small_agent()
    reference.load_state_dict(actual.state_dict())
    actual._data_batch_size = 3  # Exercise a partial last chunk: 3 + 3 + 2.
    observations = torch.randn(8, 2370)
    task_reward = torch.linspace(-1., 1., 8)
    calls = []
    for agent in (actual, reference):
        agent._exp_buffer.set_data_flat("disc_obs", observations)
        agent._exp_buffer.set_data_flat("reward", task_reward)
        agent._disc_replay_samples = 3
        cursor = [0]
        def demo(n, cursor=cursor, agent=agent):
            if agent is actual:
                calls.append(n)
            # Fixed sample stream makes chunking comparable without relying on
            # the native random sampler consuming RNG in the same call order.
            value = torch.arange(cursor[0], cursor[0]+n*2370).float().reshape(n, 2370)/20000
            cursor[0] += n*2370
            return value
        agent._env.fetch_disc_obs_demo = demo
    actual._record_disc_demo_data()
    AMPAgent._record_disc_demo_data(reference)
    assert calls == [3, 3, 2]
    torch.testing.assert_close(actual._exp_buffer.get_data_flat("disc_obs_demo"),
                               reference._exp_buffer.get_data_flat("disc_obs_demo"))
    for agent in (actual, reference):
        assert agent._disc_obs_norm._new_count == 8
        agent._disc_obs_norm.update()
    torch.testing.assert_close(actual._disc_obs_norm.get_mean(), reference._disc_obs_norm.get_mean())
    torch.testing.assert_close(actual._disc_obs_norm.get_std(), reference._disc_obs_norm.get_std(), atol=1e-6, rtol=1e-5)
    actual_reward_info = actual._compute_rewards()
    expected_reward_info = AMPAgent._compute_rewards(reference)
    torch.testing.assert_close(actual._exp_buffer.get_data_flat("reward"), reference._exp_buffer.get_data_flat("reward"))
    for key in actual_reward_info:
        torch.testing.assert_close(actual_reward_info[key], expected_reward_info[key])
    # Fill replay, then exercise its full-buffer replacement/wraparound path.
    for iteration in range(3):
        torch.manual_seed(80+iteration)
        actual._store_disc_replay_data()
        torch.manual_seed(80+iteration)
        AMPAgent._store_disc_replay_data(reference)
        assert actual._disc_buffer.get_total_samples() == reference._disc_buffer.get_total_samples()
        assert actual._disc_buffer._buffer_head == reference._disc_buffer._buffer_head
        torch.testing.assert_close(actual._disc_buffer.get_data_flat("disc_obs"),
                                   reference._disc_buffer.get_data_flat("disc_obs"))


def test_staged_buffer_keeps_native_shuffle_and_only_loads_requested_fields():
    torch.manual_seed(53)
    actual = StagedExperienceBuffer(4, 2, "cpu", host_keys=("disc_obs_demo",))
    torch.manual_seed(53)
    reference = ExperienceBuffer(4, 2, "cpu")
    for buffer in (actual, reference):
        buffer.set_data_flat("obs", torch.arange(8).float().unsqueeze(1))
        buffer.set_data_flat("disc_obs_demo", torch.arange(8).float().unsqueeze(1)+100)
        buffer._total_samples = 8
    for _ in range(2):
        with actual.sampling_fields(("obs",)):
            sample = actual.sample(3)
        expected = reference.sample(3)
        assert set(sample) == {"obs"}
        torch.testing.assert_close(sample["obs"], expected["obs"])
    # The scope restores full sampling, including host-backed demonstrations.
    sample, expected = actual.sample(2), reference.sample(2)
    for key in expected:
        torch.testing.assert_close(sample[key], expected[key])


def test_replay_sampling_plan_matches_one_native_logical_batch():
    actual = StagedExperienceBuffer(8, 1, "cpu")
    reference = ExperienceBuffer(8, 1, "cpu")
    for buffer in (actual, reference):
        buffer.push({"disc_obs": torch.arange(8).float().reshape(8, 1, 1)})
        buffer._sample_buf[:] = torch.arange(8)
        buffer._sample_buf_head = 0
    expected = reference.sample(7)["disc_obs"]
    with actual.sampling_plan(7):
        result = torch.cat([actual.sample(n)["disc_obs"] for n in (3, 3, 1)])
    torch.testing.assert_close(result, expected)
    assert actual._sample_buf_head == reference._sample_buf_head
    assert actual._sample_plan is None
    with pytest.raises(RuntimeError, match="Incomplete"):
        with actual.sampling_plan(1):
            pass
    assert actual._sample_plan is None


def test_initial_replay_larger_than_native_capacity_is_inserted_in_bounded_chunks():
    agent = small_agent()
    agent._data_batch_size = 3
    agent._disc_buffer = StagedExperienceBuffer(5, 1, "cpu")
    observations = torch.arange(8).float().unsqueeze(-1)
    agent._exp_buffer.set_data_flat("disc_obs", observations)
    torch.manual_seed(19)
    order = torch.randperm(8)
    expected = torch.empty(5, 1)
    for i, index in enumerate(order):
        expected[i % 5] = observations[index]
    torch.manual_seed(19)
    agent._store_disc_replay_data()
    assert agent._disc_buffer.get_capacity() == 5
    assert agent._disc_buffer.get_total_samples() == 8
    assert agent._disc_buffer._buffer_head == 3
    torch.testing.assert_close(agent._disc_buffer.get_data_flat("disc_obs"), expected)


@pytest.mark.parametrize("role", ["actor", "critic", "disc"])
def test_accumulated_updates_match_native_gradients_momentum_decay_and_clipping(role):
    full, chunked = small_agent(), small_agent()
    chunked.load_state_dict(full.state_dict())
    full._optimizer_microbatch_size = {}
    chunked._optimizer_microbatch_size = {role: 3}  # 7-sample batch = 3 + 3 + 1.
    obs = torch.randn(8, 255)*.1
    with torch.no_grad():
        dist = full._model.eval_actor(full._obs_norm.normalize(obs))
        norm_actions = dist.sample()
        actions = full._a_norm.unnormalize(norm_actions)
        logp = dist.log_prob(norm_actions)
    data = dict(obs=obs, action=actions, a_logp=logp, adv=torch.linspace(-1., 1., 8),
                rand_action_mask=torch.ones(8), tar_val=torch.linspace(1., 2., 8),
                disc_obs=torch.randn(8, 2370)*.1, disc_obs_demo=torch.randn(8, 2370)*.2)
    replay = torch.randn(16, 1, 2370)*.1
    for agent in (full, chunked):
        for name, value in data.items():
            agent._exp_buffer.set_data_flat(name, value)
        agent._exp_buffer._total_samples = 8
        agent._exp_buffer._sample_buf[:] = torch.arange(8)
        agent._exp_buffer._sample_buf_head = 0
        agent._disc_buffer.push({"disc_obs": replay})
        agent._disc_buffer._sample_buf[:] = torch.arange(16)
        agent._disc_buffer._sample_buf_head = 0
        optimizer = getattr(agent, f"_{role}_optimizer")
        optimizer._grad_clip = .05
        optimizer._optimizer.param_groups[0]["weight_decay"] = .01
    torch.manual_seed(29)
    expected = getattr(full, f"_update_{role}")(7, 3)
    torch.manual_seed(29)
    actual = getattr(chunked, f"_update_{role}")(7, 3)
    assert set(actual) == set(expected)
    for key in actual:
        torch.testing.assert_close(actual[key], expected[key], atol=2e-5, rtol=2e-4)
    for key, value in full._model.state_dict().items():
        torch.testing.assert_close(chunked._model.state_dict()[key], value, atol=5e-7, rtol=2e-5)
    native_optimizer, micro_optimizer = (getattr(agent, f"_{role}_optimizer") for agent in (full, chunked))
    assert native_optimizer.get_steps() == micro_optimizer.get_steps() == 3
    for first, second in zip(native_optimizer._param_list, micro_optimizer._param_list):
        torch.testing.assert_close(first.grad, second.grad, atol=2e-6, rtol=2e-4)
        first_state = native_optimizer._optimizer.state[first]
        second_state = micro_optimizer._optimizer.state[second]
        assert set(first_state) == set(second_state)
        for key in first_state:
            torch.testing.assert_close(first_state[key], second_state[key], atol=2e-6, rtol=2e-4)
    assert full._exp_buffer._sample_buf_head == chunked._exp_buffer._sample_buf_head
    assert full._disc_buffer._sample_buf_head == chunked._disc_buffer._sample_buf_head


def test_accumulated_actor_weights_ppo_by_random_actions_and_symmetry_by_all_samples():
    full, chunked = small_agent(), small_agent()
    chunked.load_state_dict(full.state_dict())
    full._optimizer_microbatch_size = {}
    chunked._optimizer_microbatch_size = {"actor": 2}
    obs = torch.randn(8, 255)*.1
    with torch.no_grad():
        dist = full._model.eval_actor(full._obs_norm.normalize(obs))
        actions = dist.sample()
        logp = dist.log_prob(actions)
    values = dict(obs=obs, action=full._a_norm.unnormalize(actions), a_logp=logp,
                  adv=torch.linspace(-1., 1., 8), rand_action_mask=torch.tensor([0., 0., 1., 0., 1., 1., 0., 1.]))
    for agent in (full, chunked):
        for key, value in values.items():
            agent._exp_buffer.set_data_flat(key, value)
        agent._exp_buffer._total_samples = 8
        agent._exp_buffer._sample_buf[:] = torch.arange(8)
        agent._exp_buffer._sample_buf_head = 0
    expected = full._update_actor(8, 1)
    actual = chunked._update_actor(8, 1)
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], atol=2e-6, rtol=2e-4)
    for key, value in full._model.state_dict().items():
        torch.testing.assert_close(chunked._model.state_dict()[key], value, atol=5e-7, rtol=2e-5)


def test_symmetry_uses_raw_current_batch_before_normalization():
    agent = small_agent()
    agent._obs_norm.record(torch.randn(8, 255)); agent._obs_norm.update()
    agent._obs_norm.set_mean_std(torch.linspace(-.3, .3, 255), torch.linspace(.5, 2., 255))
    obs = torch.randn(4, 255)*.1
    norm = agent._obs_norm.normalize(obs)
    with torch.no_grad():
        dist = agent._model.eval_actor(norm)
        action = dist.mean
        batch = dict(obs=obs, action=agent._a_norm.unnormalize(action), a_logp=dist.log_prob(action),
                     adv=torch.ones(4), rand_action_mask=torch.ones(4))
    info = agent._compute_actor_loss(batch)
    reflected = agent._model.eval_actor(agent._obs_norm.normalize(agent._env.reflect_observation(obs))).mean
    predicted = agent._model.eval_actor(norm).mean
    target = agent._a_norm.normalize(agent._env.reflect_action(agent._a_norm.unnormalize(predicted)))
    expected = 50 * (reflected-target).square().mean()
    torch.testing.assert_close(info["sym_loss"], expected.detach())
    info["actor_loss"].backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in agent._model.get_actor_params())


def test_reset_validation_detects_wrong_joint_representation_and_nan():
    model = MJCFCharModel("cpu"); model.load(str(ROOT / "data/cfg/smpl_skeletontree.xml"))
    root = torch.tensor([[0., 0., 1.]])
    q = torch.tensor([[0., 0., 0., 1.]])
    dof = torch.randn(1, 69)*.2
    position, rotation = model.forward_kinematics(root, q, model.dof_to_rot(dof))
    env = object.__new__(SmplTrackingEnv)
    env._char_ids = [0]; env._kin_char_model = model
    env._engine = SimpleNamespace(get_dof_pos=lambda _: dof, get_root_pos=lambda _: root, get_root_rot=lambda _: q,
                                  get_body_pos=lambda _: position, get_body_rot=lambda _: rotation)
    assert env.validate_reset()["max_body_position_error_m"] == 0
    position = position.clone(); position[:, -1, 0] += .2
    with pytest.raises(RuntimeError, match="mismatch"):
        env.validate_reset()
    position[:, -1, 0] = float("nan")
    with pytest.raises(RuntimeError, match="Non-finite"):
        env.validate_reset()


def test_pure_amp_emits_no_task_or_power_reward_and_needs_no_trajectory():
    env = object.__new__(SmplTrackingEnv)
    env._tracking = False
    env._char_ids = [0]
    env._task = {"power_coefficient": 0.}
    env._engine = SimpleNamespace(get_dof_forces=lambda _: torch.ones(2, 69),
                                  get_dof_vel=lambda _: torch.ones(2, 69))
    env._raw_power = torch.zeros(2)
    env._position_error = torch.ones(2)
    env._task_r = torch.ones(2)
    env._reward_buf = torch.ones(2)
    env._update_reward()
    torch.testing.assert_close(env._raw_power, torch.full((2,), 69.))
    for value in (env._position_error, env._task_r, env._reward_buf):
        torch.testing.assert_close(value, torch.zeros(2))


def test_pure_amp_can_apply_an_explicit_power_regularizer_without_a_trajectory():
    env = object.__new__(SmplTrackingEnv)
    env._tracking = False
    env._char_ids = [0]
    env._task = {"power_coefficient": 5e-5}
    env._engine = SimpleNamespace(get_dof_forces=lambda _: torch.ones(2, 69),
                                  get_dof_vel=lambda _: torch.full((2, 69), 2.))
    env._raw_power = torch.zeros(2)
    env._position_error = torch.ones(2)
    env._task_r = torch.ones(2)
    env._reward_buf = torch.zeros(2)
    env._update_reward()
    torch.testing.assert_close(env._raw_power, torch.full((2,), 138.))
    torch.testing.assert_close(env._position_error, torch.zeros(2))
    torch.testing.assert_close(env._task_r, torch.zeros(2))
    torch.testing.assert_close(env._reward_buf, torch.full((2,), -5e-5*138.))


def test_tracking_reward_and_power_cost_use_absolute_joint_mechanical_power():
    env = object.__new__(SmplTrackingEnv)
    env._tracking = True
    env._char_ids = [0]
    env._task = {"position_scale": 2., "power_coefficient": .0005}
    env._env_ids = torch.arange(2)
    env._time_buf = torch.zeros(2)
    root = torch.zeros(2, 3)
    target = torch.tensor([[0., 0., 0.], [1., 0., 0.]])
    env._engine = SimpleNamespace(
        get_dof_forces=lambda _: torch.ones(2, 69),
        get_dof_vel=lambda _: torch.full((2, 69), 2.),
        get_root_pos=lambda _: root,
    )
    env._trajectory = SimpleNamespace(position=lambda ids, times: target)
    env._raw_power = torch.zeros(2)
    env._position_error = torch.zeros(2)
    env._task_r = torch.zeros(2)
    env._reward_buf = torch.zeros(2)
    env._update_reward()
    torch.testing.assert_close(env._position_error, torch.tensor([0., 1.]))
    torch.testing.assert_close(env._task_r, torch.tensor([1., math.exp(-2.)]))
    torch.testing.assert_close(env._raw_power, torch.full((2,), 138.))
    torch.testing.assert_close(env._reward_buf, env._task_r - .0005 * env._raw_power)


def test_mechanical_power_has_no_dt_or_signed_cancellation():
    torque = torch.tensor([[2., -3., 4.]])
    velocity = torch.tensor([[-5., -7., 0.5]])
    torch.testing.assert_close(mechanical_power(torque, velocity), torch.tensor([33.]))


def test_latest_checkpoint_ignores_intermediate_and_non_training_directories(tmp_path):
    old = tmp_path / "old"; old.mkdir()
    newest = tmp_path / "new"; newest.mkdir()
    ignored = tmp_path / "play"; ignored.mkdir()
    for folder in (old, newest):
        (folder / "params").mkdir()
        (folder / "params/manifest.json").write_text("{}")
    (old / "model.pt").write_bytes(b"old")
    (newest / "model.pt").write_bytes(b"new")
    (ignored / "model.pt").write_bytes(b"ignored")
    old_time = 1_000_000_000
    import os
    os.utime(old / "model.pt", ns=(old_time, old_time))
    os.utime(newest / "model.pt", ns=(old_time + 1, old_time + 1))
    assert latest_checkpoint(tmp_path) == newest / "model.pt"


def test_trajectory_failure_is_a_strict_four_meter_termination_not_a_reward_term():
    error = torch.tensor([3.999, 4., 4.001])
    torch.testing.assert_close(
        trajectory_failure(error, 4.), torch.tensor([False, False, True]))
    torch.testing.assert_close(
        trajectory_failure(error, 4., enabled=False), torch.zeros(3, dtype=torch.bool))


def test_preflight_rejects_mixed_models_and_old_motion_cache(tmp_path):
    import hashlib
    rig = tmp_path / "rig"; rig.mkdir()
    xml = rig / "smpl.xml"; xml.write_text("skeleton")
    usd = rig / "smpl.usd"; usd.write_bytes(b"usd")
    cache = tmp_path / "motions"; cache.mkdir()
    motions = cache / "motions.yaml"; motions.write_text("motions: []\n")
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (xml, usd)}
    (rig / "asset-manifest.json").write_text(json.dumps(dict(sha256=hashes)))
    metadata = dict(skeleton_sha256=hashes["smpl.xml"], control_dt=1/30, history_steps=10)
    (cache / "manifest.json").write_text(json.dumps(metadata))
    env = dict(char_file=str(xml), motion_file=str(motions), num_disc_obs_steps=10)
    engine = dict(smpl_usd=str(usd), control_freq=30)
    resolve_paths(env.copy(), engine.copy())
    metadata["skeleton_sha256"] = "legacy-model-hash"
    (cache / "manifest.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="different SMPL skeleton"):
        resolve_paths(env.copy(), engine.copy())
    usd.write_bytes(b"other model")
    with pytest.raises(ValueError, match="official asset pair"):
        resolve_paths(env.copy(), engine.copy())


def test_official_asset_installer_selects_smpl_and_preserves_existing_files(tmp_path):
    import importlib.util
    import zipfile
    spec = importlib.util.spec_from_file_location("install_mimickit_assets", ROOT / "scripts/mimickit_amp/install_assets.py")
    installer = importlib.util.module_from_spec(spec); spec.loader.exec_module(installer)
    archive = tmp_path / "official.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        for name in ("smpl.xml", "smpl.usd", "LICENSE.txt"):
            bundle.writestr("MimicKit_Data/assets/smpl/"+name, name)
        bundle.writestr("MimicKit_Data/assets/other/large_model.usd", "unrelated")
    target = tmp_path / "installed"
    manifest = installer.install(archive, target)
    assert set(manifest["sha256"]) == {"smpl.xml", "smpl.usd", "LICENSE.txt"}
    assert {p.name for p in target.iterdir()} == {"smpl.xml", "smpl.usd", "LICENSE.txt", "asset-manifest.json"}
    with pytest.raises(FileExistsError):
        installer.install(archive, target)
    assert (target / "smpl.usd").read_text() == "smpl.usd"


def test_official_baseline_installer_extracts_only_smpl_walk_and_refuses_overwrite(tmp_path):
    import importlib.util
    import zipfile
    script = ROOT / "scripts/mimickit_amp/install_official_smpl_baseline.py"
    spec = importlib.util.spec_from_file_location("install_official_smpl_baseline", script)
    installer = importlib.util.module_from_spec(spec); spec.loader.exec_module(installer)
    archive = tmp_path / "MimicKit_Data.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("MimicKit_Data/data/motions/smpl/smpl_walk.pkl", b"official-motion")
        bundle.writestr("MimicKit_Data/data/motions/humanoid/unrelated.pkl", b"unrelated")
    destination = tmp_path / "data/motions/smpl/smpl_walk.pkl"
    result = installer.install(archive, destination)
    assert destination.read_bytes() == b"official-motion"
    assert result["archive_member"].endswith("data/motions/smpl/smpl_walk.pkl")
    assert result["sha256"] == hashlib.sha256(b"official-motion").hexdigest()
    assert json.loads(destination.with_name("official-install.json").read_text())["sha256"] == result["sha256"]
    assert installer.install(archive, destination)["sha256"] == result["sha256"]
    other = tmp_path / "other.zip"
    with zipfile.ZipFile(other, "w") as bundle:
        bundle.writestr("data/motions/smpl/smpl_walk.pkl", b"different")
    with pytest.raises(FileExistsError, match="left unchanged"):
        installer.install(other, destination)


def test_official_baseline_installer_can_install_a_named_smpl_motion(tmp_path):
    import importlib.util
    import zipfile
    script = ROOT / "scripts/mimickit_amp/install_official_smpl_baseline.py"
    spec = importlib.util.spec_from_file_location("install_official_smpl_run", script)
    installer = importlib.util.module_from_spec(spec); spec.loader.exec_module(installer)
    archive = tmp_path / "MimicKit_Data.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("MimicKit_Data/motions/smpl/smpl_walk.pkl", b"walk")
        bundle.writestr("MimicKit_Data/motions/smpl/smpl_run.pkl", b"run")
    destination = tmp_path / "data/motions/smpl/smpl_run.pkl"
    result = installer.install(archive, destination, "smpl_run.pkl")
    assert destination.read_bytes() == b"run"
    assert result["archive_member"].endswith("motions/smpl/smpl_run.pkl")
    manifest = destination.with_name("official-smpl_run-install.json")
    assert json.loads(manifest.read_text())["sha256"] == hashlib.sha256(b"run").hexdigest()


def test_pure_amp_restart_and_explicit_tracking_transfer(tmp_path):
    import yaml
    paths = [ROOT / f"data/cfg/mimickit_amp/{name}.yaml" for name in ("env", "agent", "engine")]
    baseline = run_configurations(paths, amp_only=True)
    env, agent, _ = baseline
    assert not env["tracking"]["enabled"]
    assert env["tracking"]["power_coefficient"] == 0
    assert agent["task_reward_weight"] == 0
    assert agent["symmetry_weight"] == 2. and agent["symmetry_scale"] == 50.
    assert agent["disc_reward_weight"] == 1
    assert env["global_obs"] is True and env["episode_length"] == 10
    powered_env, powered_agent, _ = run_configurations(
        paths, amp_only=True, power_coefficient=5e-5
    )
    assert not powered_env["tracking"]["enabled"]
    assert powered_env["tracking"]["power_coefficient"] == 5e-5
    assert powered_agent["task_reward_weight"] == 1
    assert powered_agent["disc_reward_weight"] == 1
    for name, cfg in zip(("env", "agent", "engine"), baseline):
        (tmp_path / f"{name}.yaml").write_text(yaml.safe_dump(cfg))
    # Loading a saved baseline alone must not accidentally enable a new task.
    restart = run_configurations(paths, checkpoint_params=tmp_path)
    assert not restart[0]["tracking"]["enabled"] and restart[1]["task_reward_weight"] == 0
    assert restart[1]["symmetry_weight"] == 2.
    changed_data = run_configurations(paths, checkpoint_params=tmp_path, motion_file="weighted-motions.yaml")
    assert changed_data[0]["motion_file"] == "weighted-motions.yaml"
    assert not changed_data[0]["tracking"]["enabled"]
    assert changed_data[1]["task_reward_weight"] == 0 and changed_data[1]["symmetry_weight"] == 2.
    # Explicit transfer loads CURRENT configs, rather than silently retaining task=0.
    tracking = run_configurations(paths, checkpoint_params=tmp_path, finetune_tracking=True)
    assert tracking[0]["tracking"]["enabled"]
    assert tracking[0]["tracking"]["power_coefficient"] == 0
    assert tracking[1]["task_reward_weight"] == .5 and tracking[1]["disc_reward_weight"] == .5
    assert tracking[1]["symmetry_weight"] == 2.
    # Native MimicKit stores the same files with a ``_config`` suffix.
    official = tmp_path / "official"; official.mkdir()
    for name, cfg in zip(("env", "agent", "engine"), baseline):
        (official / f"{name}_config.yaml").write_text(yaml.safe_dump(cfg))
    official_tracking = run_configurations(paths, checkpoint_params=official, finetune_tracking=True)
    assert official_tracking[0]["tracking"]["enabled"]
    assert official_tracking[0]["tracking"]["power_coefficient"] == 0
    from_scratch_paths = [ROOT / "data/cfg/mimickit_amp/env_tracking.yaml", *paths[1:]]
    from_scratch = run_configurations(from_scratch_paths)
    assert from_scratch[0]["tracking"]["enabled"]
    assert from_scratch[1]["task_reward_weight"] == from_scratch[1]["disc_reward_weight"] == .5
    # Preserve the semantic coordinate convention of older local-observation
    # checkpoints, even though it has the same dimension as global observations.
    env["global_obs"] = False
    (tmp_path / "env.yaml").write_text(yaml.safe_dump(env))
    older_transfer = run_configurations(paths, checkpoint_params=tmp_path, finetune_tracking=True)
    assert older_transfer[0]["global_obs"] is False
    with pytest.raises(ValueError, match="mutually exclusive"):
        run_configurations(paths, checkpoint_params=tmp_path, amp_only=True, finetune_tracking=True)
    with pytest.raises(ValueError, match="requires a checkpoint"):
        run_configurations(paths, finetune_tracking=True)
    with pytest.raises(ValueError, match="requires amp_only"):
        run_configurations(paths, power_coefficient=5e-5)
    with pytest.raises(ValueError, match="finite and non-negative"):
        run_configurations(paths, amp_only=True, power_coefficient=float("nan"))
    agent["symmetry_weight"] = 0
    (tmp_path / "agent.yaml").write_text(yaml.safe_dump(agent))
    with pytest.raises(ValueError, match="positive symmetry_weight"):
        run_configurations(paths, checkpoint_params=tmp_path, amp_only=True)


def test_tracking_transfer_preserves_initial_actions_and_critic_with_new_goals():
    agent = small_agent()
    samples = torch.randn(32, 255); samples[:, -20:] = 0
    agent._obs_norm.record(samples); agent._obs_norm.update()
    old_mean, old_std = agent._obs_norm.get_mean().clone(), agent._obs_norm.get_std().clone()
    obs = torch.randn(8, 255); obs[:, -20:] = 0
    with torch.no_grad():
        old_action = agent._model.eval_actor(agent._obs_norm.normalize(obs)).mean.clone()
        old_value = agent._model.eval_critic(agent._obs_norm.normalize(obs)).clone()
    before = {k: v.clone() for k, v in agent._model.state_dict().items()}
    agent.initialize_tracking_inputs()
    obs[:, -20:] = torch.randn(8, 20)*3  # a newly enabled path must not kick the loaded controller
    with torch.no_grad():
        new_action = agent._model.eval_actor(agent._obs_norm.normalize(obs)).mean
        new_value = agent._model.eval_critic(agent._obs_norm.normalize(obs))
    torch.testing.assert_close(old_action, new_action)
    torch.testing.assert_close(old_value, new_value)
    torch.testing.assert_close(agent._obs_norm.get_mean()[:-20], old_mean[:-20])
    torch.testing.assert_close(agent._obs_norm.get_std()[:-20], old_std[:-20])
    torch.testing.assert_close(agent._obs_norm.get_std()[-20:], torch.ones(20))
    for key, value in agent._model.state_dict().items():
        if key in ("_actor_layers.0.weight", "_critic_layers.0.weight"):
            torch.testing.assert_close(value[:, :-20], before[key][:, :-20])
        else:
            torch.testing.assert_close(value, before[key])
    # Native normalizer updates remain finite after the selective initialization.
    agent._obs_norm.record(obs); agent._obs_norm.update()
    assert torch.isfinite(agent._obs_norm.normalize(obs)).all()


def test_official_235_feature_locomotion_checkpoint_expands_for_tracking(tmp_path):
    source = small_agent()
    source.initialize_tracking_inputs()
    char_obs = torch.randn(8, 235)
    zero_goals = torch.zeros(8, 20)
    base_obs = torch.cat((char_obs, zero_goals), dim=-1)
    with torch.no_grad():
        expected_action = source._model.eval_actor(source._obs_norm.normalize(base_obs)).mean.clone()
        expected_value = source._model.eval_critic(source._obs_norm.normalize(base_obs)).clone()

    state = source.state_dict()
    state["_obs_norm._mean"] = state["_obs_norm._mean"][:235].clone()
    state["_obs_norm._std"] = state["_obs_norm._std"][:235].clone()
    for key in ("_model._actor_layers.0.weight", "_model._critic_layers.0.weight"):
        state[key] = state[key][:, :235].clone()
    checkpoint = tmp_path / "official.pt"
    torch.save(state, checkpoint)

    target = small_agent()
    target.load_locomotion_for_tracking(str(checkpoint))
    obs = torch.cat((char_obs, torch.randn(8, 20) * 5), dim=-1)
    with torch.no_grad():
        actual_action = target._model.eval_actor(target._obs_norm.normalize(obs)).mean
        actual_value = target._model.eval_critic(target._obs_norm.normalize(obs))
    torch.testing.assert_close(actual_action, expected_action)
    torch.testing.assert_close(actual_value, expected_value)


def test_tensorboard_is_visible_at_startup_and_writes_each_iteration_once(tmp_path):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    from socnav_sim.mimickit_amp.logging import IterationTBLogger
    logger = IterationTBLogger()
    logger.set_step_key("Samples")
    logger.configure_output_file(str(tmp_path / "log.txt"))
    events = EventAccumulator(str(tmp_path)).Reload()
    assert events.Scalars("Progress/Started")[0].value == 1
    for i in range(3):
        logger.log("Iteration", i)
        logger.log("Samples", (i+1)*32)
        logger.log("Actor_Loss", float(i))
        logger.print_log()  # every iteration, including non-checkpoint iterations
        logger.write_log()  # upstream also calls this at checkpoint iterations
    events.Reload()
    scalars = events.Scalars("Misc/Actor_Loss")
    assert [event.step for event in scalars] == [32, 64, 96]
    assert [event.value for event in scalars] == [0., 1., 2.]
    assert logger._row_count == 3
    logger._writer.close()
