"""Round trip regression test on a released PhysTwin checkpoint (needs CUDA).

Skipped unless a GPU is available and PHYSTWIN_CASE names a case whose
released data is unpacked in the repo root (data/, experiments/ and
experiments_optimization/ from the README). Run from the repo root:

    PHYSTWIN_CASE=double_stretch_sloth pytest tests/test_checkpoint_roundtrip_gpu.py

It checks three things against the trajectory the released (legacy)
checkpoint produces today:
  1. the same checkpoint with the spring topology added reproduces it exactly
  2. a checkpoint whose springs, rest lengths and spring_Y are all stored in a
     shuffled order reproduces it up to float summation order, which only works
     if loading really uses the saved springs
  3. legacy checkpoints still take the old path unchanged, including on a
     trainer that has already loaded a checkpoint with different springs
"""

import glob
import json
import os
import pickle

import numpy as np
import pytest
import torch

CASE = os.environ.get("PHYSTWIN_CASE")
BASE_PATH = os.environ.get("PHYSTWIN_DATA", "./data/different_types")

pytestmark = pytest.mark.skipif(
    CASE is None or not torch.cuda.is_available(),
    reason="needs CUDA and PHYSTWIN_CASE pointing at a released case",
)


def configure(cfg, case_name, base_path):
    # Same setup as inference_warp.py
    if "cloth" in case_name or "package" in case_name:
        cfg.load_from_yaml("configs/cloth.yaml")
    else:
        cfg.load_from_yaml("configs/real.yaml")
    with open(f"experiments_optimization/{case_name}/optimal_params.pkl", "rb") as f:
        cfg.set_optimal_params(pickle.load(f))
    with open(f"{base_path}/{case_name}/calibrate.pkl", "rb") as f:
        c2ws = pickle.load(f)
    cfg.c2ws = np.array(c2ws)
    cfg.w2cs = np.array([np.linalg.inv(c2w) for c2w in c2ws])
    with open(f"{base_path}/{case_name}/metadata.json", "r") as f:
        meta = json.load(f)
    cfg.intrinsics = np.array(meta["intrinsics"])
    cfg.WH = meta["WH"]
    cfg.overlay_path = f"{base_path}/{case_name}/color"


def load_like_test(trainer, model_path):
    # The loading block shared by test(), interactive_playground(),
    # visualize_force() and visualize_material()
    checkpoint = torch.load(model_path, map_location="cuda:0")
    trainer._load_checkpoint_springs(checkpoint)
    spring_Y = checkpoint["spring_Y"]
    assert len(spring_Y) == trainer.simulator.n_springs
    trainer.simulator.set_spring_Y(torch.log(spring_Y).detach().clone())
    trainer.simulator.set_collide(
        checkpoint["collide_elas"].detach().clone(),
        checkpoint["collide_fric"].detach().clone(),
    )
    trainer.simulator.set_collide_object(
        checkpoint["collide_object_elas"].detach().clone(),
        checkpoint["collide_object_fric"].detach().clone(),
    )
    return checkpoint


def rollout(trainer):
    # visualize_sim() without the video, returns the (frames, points, 3) trajectory
    import warp as wp
    from qqtt.utils import cfg

    sim = trainer.simulator
    sim.set_init_state(sim.wp_init_vertices, sim.wp_init_velocities)
    frames = [wp.to_torch(sim.wp_states[0].wp_x, requires_grad=False).cpu()]
    for i in range(1, trainer.dataset.frame_len):
        if cfg.data_type == "real":
            sim.set_controller_target(i, pure_inference=True)
        if sim.object_collision_flag:
            sim.update_collision_graph()
        if cfg.use_graph:
            wp.capture_launch(sim.forward_graph)
        else:
            sim.step()
        frames.append(wp.to_torch(sim.wp_states[-1].wp_x, requires_grad=False).cpu())
        sim.set_init_state(sim.wp_states[-1].wp_x, sim.wp_states[-1].wp_v)
    return torch.stack(frames).numpy()


def run_case(model_path, tmp_path):
    from qqtt import InvPhyTrainerWarp
    from qqtt.utils import cfg

    configure(cfg, CASE, BASE_PATH)
    trainer = InvPhyTrainerWarp(
        data_path=f"{BASE_PATH}/{CASE}/final_data.pkl",
        base_dir=str(tmp_path / "run"),
        pure_inference_mode=True,
    )
    checkpoint = load_like_test(trainer, model_path)
    return trainer, checkpoint, rollout(trainer)


def test_released_checkpoint_round_trip(tmp_path):
    from qqtt.utils.spring_io import spring_topology_state

    released = sorted(glob.glob(f"experiments/{CASE}/train/best_*.pth"))
    assert released, f"no released checkpoint for {CASE}"

    # Legacy path: what the released checkpoint does today
    trainer, legacy, reference = run_case(released[0], tmp_path)
    assert "springs" not in legacy, "expected a checkpoint from before this change"
    springs = trainer.init_springs.cpu().clone()
    rest_lengths = trainer.init_rest_lengths.cpu().clone()

    # 1. Same checkpoint plus its topology: identical trajectory
    with_topology = dict(legacy)
    with_topology.update(spring_topology_state(springs, rest_lengths))
    path = tmp_path / "with_topology.pth"
    torch.save(with_topology, path)
    _, _, same = run_case(path, tmp_path)
    np.testing.assert_allclose(same, reference, atol=1e-5)

    # 2. Stored in a shuffled order: the rebuilt springs no longer line up with
    # spring_Y by position, so this only matches if the saved springs are used.
    # Object springs stay first so num_object_springs remains valid.
    n_obj = legacy["num_object_springs"]
    g = torch.Generator().manual_seed(0)
    perm = torch.cat(
        [
            torch.randperm(n_obj, generator=g),
            n_obj + torch.randperm(len(springs) - n_obj, generator=g),
        ]
    )
    shuffled = dict(legacy)
    shuffled["spring_Y"] = legacy["spring_Y"].cpu()[perm]
    shuffled.update(spring_topology_state(springs[perm], rest_lengths[perm]))
    path = tmp_path / "shuffled.pth"
    torch.save(shuffled, path)
    trainer_s, _, shuffled_traj = run_case(path, tmp_path)
    assert torch.equal(trainer_s.init_springs.cpu(), springs[perm])
    # Warp's atomic adds are not order deterministic on GPU, so allow float noise
    np.testing.assert_allclose(shuffled_traj, reference, atol=1e-3)

    # 3. The same trainer then loads the legacy checkpoint: it must go back to
    # the springs rebuilt from the config, not keep the shuffled ones it loaded
    # last, and reproduce the reference exactly
    load_like_test(trainer_s, released[0])
    assert torch.equal(trainer_s.init_springs.cpu(), springs)
    assert torch.equal(trainer_s.init_rest_lengths.cpu(), rest_lengths)
    np.testing.assert_allclose(rollout(trainer_s), reference, atol=1e-5)
