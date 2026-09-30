"""CPU tests of the Warp spring mass simulator.

These cover the spring force kernel against Hooke's law, and check at the
level of a simulated trajectory that the stiffness has to stay attached to
the spring it was learned for, which is what saving the spring topology in
checkpoints guarantees.
"""

import numpy as np
import pytest
import torch

wp = pytest.importorskip("warp")

Y_MIN, Y_MAX = 1e3, 1e5


def spring_forces(
    sim,
    x,
    springs,
    rest_lengths,
    spring_Y,
    num_object_points=None,
    control_x=None,
    v=None,
    dashpot=0.0,
):
    """Run eval_springs once on CPU and return the per point forces."""
    x = np.asarray(x, dtype=np.float32)
    n = len(x) if num_object_points is None else num_object_points
    v = np.zeros_like(x) if v is None else np.asarray(v, dtype=np.float32)
    control_x = np.zeros((1, 3), np.float32) if control_x is None else control_x
    control_x = np.asarray(control_x, dtype=np.float32)
    f = wp.zeros(n, dtype=wp.vec3, device="cpu")
    wp.launch(
        sim.eval_springs,
        dim=len(springs),
        inputs=[
            wp.array(x, dtype=wp.vec3, device="cpu"),
            wp.array(v, dtype=wp.vec3, device="cpu"),
            wp.array(control_x, dtype=wp.vec3, device="cpu"),
            wp.array(np.zeros_like(control_x), dtype=wp.vec3, device="cpu"),
            n,
            wp.array(np.asarray(springs, dtype=np.int32), dtype=wp.vec2i, device="cpu"),
            wp.array(
                np.asarray(rest_lengths, dtype=np.float32), dtype=float, device="cpu"
            ),
            # the simulator stores log stiffness
            wp.array(
                np.log(np.asarray(spring_Y, dtype=np.float32)),
                dtype=float,
                device="cpu",
            ),
            dashpot,
            Y_MIN,
            Y_MAX,
        ],
        outputs=[f],
        device="cpu",
    )
    return f.numpy()


def test_single_spring_obeys_hookes_law(spring_mass_warp):
    Y, rest, length = 2e4, 1.0, 1.2
    f = spring_forces(
        spring_mass_warp,
        x=[[0, 0, 0], [length, 0, 0]],
        springs=[[0, 1]],
        rest_lengths=[rest],
        spring_Y=[Y],
    )
    expected = Y * (length / rest - 1.0)
    np.testing.assert_allclose(f[0], [expected, 0, 0], rtol=1e-5)
    np.testing.assert_allclose(f[1], [-expected, 0, 0], rtol=1e-5)


def test_compressed_spring_pushes_apart(spring_mass_warp):
    f = spring_forces(
        spring_mass_warp,
        x=[[0, 0, 0], [0, 0.5, 0]],
        springs=[[0, 1]],
        rest_lengths=[1.0],
        spring_Y=[1e4],
    )
    assert f[0][1] < 0 and f[1][1] > 0


def test_stiffness_is_clamped_to_max(spring_mass_warp):
    f = spring_forces(
        spring_mass_warp,
        x=[[0, 0, 0], [1.1, 0, 0]],
        springs=[[0, 1]],
        rest_lengths=[1.0],
        spring_Y=[Y_MAX * 10],
    )
    np.testing.assert_allclose(f[0][0], Y_MAX * 0.1, rtol=1e-4)


def test_spring_below_min_stiffness_is_off(spring_mass_warp):
    f = spring_forces(
        spring_mass_warp,
        x=[[0, 0, 0], [2.0, 0, 0]],
        springs=[[0, 1]],
        rest_lengths=[1.0],
        spring_Y=[Y_MIN * 0.5],
    )
    np.testing.assert_array_equal(f, 0.0)


def test_dashpot_opposes_separation(spring_mass_warp):
    # At rest length there is no spring force, only damping
    f = spring_forces(
        spring_mass_warp,
        x=[[0, 0, 0], [1.0, 0, 0]],
        v=[[-1, 0, 0], [1, 0, 0]],
        springs=[[0, 1]],
        rest_lengths=[1.0],
        spring_Y=[1e4],
        dashpot=10.0,
    )
    np.testing.assert_allclose(f[0], [20.0, 0, 0], rtol=1e-5)
    np.testing.assert_allclose(f[1], [-20.0, 0, 0], rtol=1e-5)


def test_controller_spring_only_pushes_object_point(spring_mass_warp):
    # Index 1 is past num_object_points, so it refers to control point 0
    f = spring_forces(
        spring_mass_warp,
        x=[[0, 0, 0]],
        num_object_points=1,
        control_x=[[0, 0, 1.5]],
        springs=[[1, 0]],
        rest_lengths=[1.0],
        spring_Y=[1e4],
    )
    assert f.shape == (1, 3)
    np.testing.assert_allclose(f[0], [0, 0, 1e4 * 0.5], rtol=1e-5)


def random_cloud(n=40, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 0.1, size=(n, 3)).astype(np.float32)
    x[:, 2] += 0.5  # keep it clear of the ground
    springs = []
    for i in range(n):
        d = np.linalg.norm(x - x[i], axis=1)
        for j in np.argsort(d)[1:6]:
            springs.append([i, int(j)])
    springs = np.unique(np.sort(np.array(springs), axis=1), axis=0).astype(np.int32)
    rest = np.linalg.norm(x[springs[:, 0]] - x[springs[:, 1]], axis=1)
    spring_Y = rng.uniform(2e3, 5e4, size=len(springs)).astype(np.float32)
    return x, springs, rest.astype(np.float32), spring_Y


def simulate(sim_module, x, springs, rest, spring_Y, frames=20):
    """Build a SpringMassSystemWarp on CPU and roll it out, as visualize_sim does."""
    sim_module.cfg.use_graph = False
    sim_module.cfg.collision_learn = True
    sim_module.cfg.data_type = "synthetic"
    sim_module.cfg.device = "cpu"

    vertices = torch.tensor(x)
    sim = sim_module.SpringMassSystemWarp(
        vertices,
        torch.tensor(springs, dtype=torch.int32),
        torch.tensor(rest),
        torch.ones(len(x)),
        dt=5e-5,
        num_substeps=50,
        spring_Y=3e4,
        collide_elas=0.5,
        collide_fric=0.3,
        dashpot_damping=100,
        drag_damping=3,
        num_object_points=len(x),
        spring_Y_min=Y_MIN,
        spring_Y_max=Y_MAX,
        gt_object_points=vertices[None].repeat(2, 1, 1),
    )
    sim.set_spring_Y(torch.log(torch.tensor(spring_Y)))

    # Start from a stretched shape so the springs do real work
    wp_x0 = wp.array(x * 1.2, dtype=wp.vec3, device="cpu")
    sim.set_init_state(wp_x0, sim.wp_init_velocities, pure_inference=True)
    for _ in range(frames):
        sim.step()
        sim.set_init_state(
            sim.wp_states[-1].wp_x, sim.wp_states[-1].wp_v, pure_inference=True
        )
    return sim.wp_states[-1].wp_x.numpy().copy()


def test_simulation_moves_and_springs_contract(spring_mass_warp):
    x, springs, rest, spring_Y = random_cloud()
    out = simulate(spring_mass_warp, x, springs, rest, spring_Y)
    stretched = np.linalg.norm(
        (x * 1.2)[springs[:, 0]] - (x * 1.2)[springs[:, 1]], axis=1
    )
    after = np.linalg.norm(out[springs[:, 0]] - out[springs[:, 1]], axis=1)
    assert np.all(np.isfinite(out))
    assert np.mean(np.abs(after - rest)) < np.mean(np.abs(stretched - rest))


def test_stiffness_travels_with_its_spring(spring_mass_warp, spring_io):
    """End to end on CPU: learned stiffness must stay keyed to its point pair.

    trained:  the springs and stiffness as saved at training time
    permuted: the same springs in a different order, stiffness permuted with
              them, which must give the same motion
    stale:    springs rebuilt in a different order but stiffness applied by
              position, which is what a legacy load silently does if the
              rebuild order ever changes
    restored: the permuted rebuild passed through resolve_spring_topology
              with a checkpoint that stores the topology
    """
    x, springs, rest, spring_Y = random_cloud()
    perm = np.random.default_rng(1).permutation(len(springs))

    trained = simulate(spring_mass_warp, x, springs, rest, spring_Y)
    permuted = simulate(spring_mass_warp, x, springs[perm], rest[perm], spring_Y[perm])
    stale = simulate(spring_mass_warp, x, springs[perm], rest[perm], spring_Y)

    ckpt = {"spring_Y": torch.tensor(spring_Y), "num_object_springs": len(springs)}
    ckpt.update(
        spring_io.spring_topology_state(torch.tensor(springs), torch.tensor(rest))
    )
    use_springs, use_rest, status = spring_io.resolve_spring_topology(
        ckpt,
        torch.tensor(springs[perm]),
        torch.tensor(rest[perm]),
        num_vertices=len(x),
        vertices=torch.tensor(x),
    )
    assert status == spring_io.SAVED
    restored = simulate(
        spring_mass_warp, x, use_springs.numpy(), use_rest.numpy(), spring_Y
    )

    # Only float summation order differs between these
    np.testing.assert_allclose(permuted, trained, atol=1e-5)
    np.testing.assert_allclose(restored, trained, atol=1e-5)
    # The stale mapping is a different object, not rounding noise
    assert np.abs(stale - trained).max() > 100 * np.abs(permuted - trained).max()
    assert np.abs(stale - trained).max() > 1e-4
