"""CPU tests for saving and restoring the spring topology in checkpoints."""

import pytest
import torch


def make_topology(num_vertices=12, num_springs=30, seed=0):
    g = torch.Generator().manual_seed(seed)
    springs = torch.randint(0, num_vertices, (num_springs, 2), generator=g)
    springs = springs.to(torch.int32)
    rest_lengths = torch.rand(num_springs, generator=g) + 0.01
    spring_Y = torch.rand(num_springs, generator=g) * 1e4 + 1e3
    return springs, rest_lengths, spring_Y


def legacy_checkpoint(spring_Y, num_object_springs):
    # The keys a checkpoint written before this change contains
    return {
        "epoch": 0,
        "num_object_springs": num_object_springs,
        "spring_Y": spring_Y,
        "collide_elas": torch.tensor([0.5]),
        "collide_fric": torch.tensor([0.3]),
        "collide_object_elas": torch.tensor([0.7]),
        "collide_object_fric": torch.tensor([0.3]),
        "optimizer_state_dict": {},
    }


def new_checkpoint(spring_io, springs, rest_lengths, spring_Y):
    ckpt = legacy_checkpoint(spring_Y, len(springs))
    ckpt.update(spring_io.spring_topology_state(springs, rest_lengths))
    return ckpt


def save_and_load(ckpt, path):
    torch.save(ckpt, path)
    return torch.load(path, map_location="cpu")


def test_round_trip_through_torch_save(spring_io, tmp_path):
    springs, rest_lengths, spring_Y = make_topology()
    loaded = save_and_load(
        new_checkpoint(spring_io, springs, rest_lengths, spring_Y),
        tmp_path / "ckpt.pth",
    )

    assert loaded["springs"].dtype == torch.int32
    assert loaded["rest_lengths"].dtype == torch.float32
    assert torch.equal(loaded["springs"], springs)
    assert torch.equal(loaded["rest_lengths"], rest_lengths)

    out_springs, out_rest, status = spring_io.resolve_spring_topology(
        loaded, springs, rest_lengths, num_vertices=12
    )
    assert status == spring_io.MATCH
    assert out_springs is springs and out_rest is rest_lengths


def test_round_trip_with_weights_only_load(spring_io, tmp_path):
    # torch >= 2.6 defaults torch.load to weights_only=True; the new keys are
    # plain tensors so they must load under that setting too
    springs, rest_lengths, spring_Y = make_topology()
    path = tmp_path / "ckpt.pth"
    torch.save(new_checkpoint(spring_io, springs, rest_lengths, spring_Y), path)
    loaded = torch.load(path, map_location="cpu", weights_only=True)
    assert torch.equal(loaded["springs"], springs)


def test_legacy_checkpoint_keeps_rebuilt_springs(spring_io, tmp_path):
    springs, rest_lengths, spring_Y = make_topology()
    loaded = save_and_load(
        legacy_checkpoint(spring_Y, len(springs)), tmp_path / "legacy.pth"
    )

    out_springs, out_rest, status = spring_io.resolve_spring_topology(
        loaded, springs, rest_lengths, num_vertices=12
    )
    assert status == spring_io.LEGACY
    assert out_springs is springs and out_rest is rest_lengths


def test_reordered_rebuild_uses_saved_springs(spring_io):
    # The case this change is for: the springs rebuilt at load time come out
    # in a different order from the ones the stiffness was learned on
    springs, rest_lengths, spring_Y = make_topology()
    ckpt = new_checkpoint(spring_io, springs, rest_lengths, spring_Y)

    perm = torch.randperm(len(springs), generator=torch.Generator().manual_seed(1))
    rebuilt_springs, rebuilt_rest = springs[perm], rest_lengths[perm]

    out_springs, out_rest, status = spring_io.resolve_spring_topology(
        ckpt, rebuilt_springs, rebuilt_rest, num_vertices=12
    )
    assert status == spring_io.SAVED
    assert torch.equal(out_springs, springs)
    assert torch.equal(out_rest, rest_lengths)

    # Stiffness per point pair is what was trained with the saved springs...
    def stiffness_by_pair(s, y):
        return {tuple(p.tolist()): float(v) for p, v in zip(s, y)}

    trained = stiffness_by_pair(springs, spring_Y)
    assert stiffness_by_pair(out_springs, ckpt["spring_Y"]) == trained
    # ...and would not be with the reordered rebuild, which is the silent error
    assert stiffness_by_pair(rebuilt_springs, ckpt["spring_Y"]) != trained


def test_rebuild_with_different_spring_count_uses_saved_springs(spring_io):
    springs, rest_lengths, spring_Y = make_topology()
    ckpt = new_checkpoint(spring_io, springs, rest_lengths, spring_Y)

    out_springs, _, status = spring_io.resolve_spring_topology(
        ckpt, springs[:-3], rest_lengths[:-3], num_vertices=12
    )
    assert status == spring_io.SAVED
    assert len(out_springs) == len(spring_Y)


def test_same_springs_different_rest_lengths_uses_saved(spring_io):
    springs, rest_lengths, spring_Y = make_topology()
    ckpt = new_checkpoint(spring_io, springs, rest_lengths, spring_Y)

    _, out_rest, status = spring_io.resolve_spring_topology(
        ckpt, springs, rest_lengths + 1e-3, num_vertices=12
    )
    assert status == spring_io.SAVED
    assert torch.equal(out_rest, rest_lengths)


def test_saved_tensors_are_detached_cpu_copies(spring_io):
    springs, rest_lengths, _ = make_topology()
    springs64 = springs.to(torch.int64)
    rest = rest_lengths.clone().requires_grad_(True)

    state = spring_io.spring_topology_state(springs64, rest)
    assert state["springs"].dtype == torch.int32
    assert state["springs"].device.type == "cpu"
    assert not state["rest_lengths"].requires_grad

    # Later in place changes to the simulator's tensors must not leak into a
    # checkpoint that has already been built
    springs64[0, 0] = 11
    assert state["springs"][0, 0] == springs[0, 0]


def test_output_matches_rebuilt_device_and_dtype(spring_io):
    springs, rest_lengths, spring_Y = make_topology()
    ckpt = new_checkpoint(spring_io, springs, rest_lengths, spring_Y)
    rebuilt = springs.flip(0).to(torch.int64)

    out_springs, out_rest, status = spring_io.resolve_spring_topology(
        ckpt, rebuilt, rest_lengths.flip(0).double(), num_vertices=12
    )
    assert status == spring_io.SAVED
    assert out_springs.dtype == torch.int64
    assert out_rest.dtype == torch.float64


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda c: c.pop("rest_lengths"), "both are needed"),
        (lambda c: c.pop("springs"), "both are needed"),
        (lambda c: c.__setitem__("springs", c["springs"][:, :1]), "shape"),
        (lambda c: c.__setitem__("rest_lengths", c["rest_lengths"][:-1]), "shape"),
        (lambda c: c.__setitem__("spring_Y", c["spring_Y"][:-1]), "spring_Y"),
    ],
)
def test_malformed_checkpoint_is_rejected(spring_io, mutate, message):
    springs, rest_lengths, spring_Y = make_topology()
    ckpt = new_checkpoint(spring_io, springs, rest_lengths, spring_Y)
    mutate(ckpt)
    with pytest.raises(ValueError, match=message):
        spring_io.resolve_spring_topology(ckpt, springs, rest_lengths, 12)


def test_springs_from_other_data_are_rejected(spring_io):
    # A checkpoint whose springs index more vertices than this data has
    springs, rest_lengths, spring_Y = make_topology(num_vertices=12)
    ckpt = new_checkpoint(spring_io, springs, rest_lengths, spring_Y)
    with pytest.raises(ValueError, match="does not belong"):
        spring_io.resolve_spring_topology(
            ckpt, springs, rest_lengths, num_vertices=int(springs.max())
        )


# Geometry checks and the immutable config topology


def make_case(num_vertices=12, seed=0, extent=0.1):
    """A small case whose rest lengths really are the spring lengths, built the
    way _init_start does it (float64 distances stored as float32)."""
    g = torch.Generator().manual_seed(seed)
    vertices = (torch.rand(num_vertices, 3, generator=g, dtype=torch.float64) * extent)
    pairs = [(i, j) for i in range(num_vertices) for j in range(i + 1, num_vertices)]
    pick = torch.randperm(len(pairs), generator=g)[:30]
    springs = torch.tensor([pairs[k] for k in pick], dtype=torch.int32)
    rest_lengths = torch.linalg.norm(
        vertices[springs[:, 0].long()] - vertices[springs[:, 1].long()], dim=1
    ).to(torch.float32)
    spring_Y = torch.rand(len(springs), generator=g) * 1e4 + 1e3
    return vertices.to(torch.float32), springs, rest_lengths, spring_Y


def test_geometry_check_accepts_same_case(spring_io):
    vertices, springs, rest_lengths, spring_Y = make_case()
    ckpt = new_checkpoint(spring_io, springs, rest_lengths, spring_Y)
    perm = torch.randperm(len(springs), generator=torch.Generator().manual_seed(1))

    out_springs, _, status = spring_io.resolve_spring_topology(
        ckpt, springs[perm], rest_lengths[perm], len(vertices), vertices=vertices
    )
    assert status == spring_io.SAVED
    assert torch.equal(out_springs, springs)


def test_other_case_with_same_vertex_count_is_rejected(spring_io):
    # Reviewer's case 1: indices are all in range, so only geometry can tell
    vertices_a, springs_a, rest_a, spring_Y_a = make_case(seed=0)
    vertices_b, springs_b, rest_b, _ = make_case(seed=1)
    assert len(vertices_a) == len(vertices_b)
    ckpt = new_checkpoint(spring_io, springs_a, rest_a, spring_Y_a)

    # Without the geometry the checkpoint would be accepted
    _, _, status = spring_io.resolve_spring_topology(
        ckpt, springs_b, rest_b, len(vertices_b)
    )
    assert status == spring_io.SAVED
    with pytest.raises(ValueError, match="do not match the current vertices"):
        spring_io.resolve_spring_topology(
            ckpt, springs_b, rest_b, len(vertices_b), vertices=vertices_b
        )


def test_geometry_check_catches_small_shift(spring_io):
    # A case whose points moved by 1 mm is different data, not rounding noise
    vertices, springs, rest_lengths, spring_Y = make_case()
    ckpt = new_checkpoint(spring_io, springs, rest_lengths, spring_Y)
    moved = vertices.clone()
    moved[3] += 1e-3
    touches_3 = (springs == 3).any(dim=1).any()
    assert touches_3
    with pytest.raises(ValueError, match="do not match"):
        spring_io.resolve_spring_topology(
            ckpt, springs, rest_lengths, len(moved), vertices=moved
        )


def test_config_topology_legacy_after_saved_uses_config(spring_io):
    # Reviewer's case 2: after a checkpoint replaces the springs, a later
    # legacy checkpoint must resolve against the config, not the replaced ones
    vertices, springs, rest_lengths, spring_Y = make_case()
    config = spring_io.ConfigSpringTopology(springs, rest_lengths, 20, vertices)

    perm = torch.randperm(len(springs), generator=torch.Generator().manual_seed(2))
    saved = new_checkpoint(spring_io, springs[perm], rest_lengths[perm], spring_Y)
    saved["num_object_springs"] = 25

    s1, r1, n1, status1 = config.resolve(saved)
    assert status1 == spring_io.SAVED
    assert torch.equal(s1, springs[perm]) and n1 == 25

    s2, r2, n2, status2 = config.resolve(legacy_checkpoint(spring_Y, 20))
    assert status2 == spring_io.LEGACY
    assert torch.equal(s2, springs) and torch.equal(r2, rest_lengths) and n2 == 20

    # A checkpoint that matches the config also resolves to the config
    match = new_checkpoint(spring_io, springs, rest_lengths, spring_Y)
    s3, _, n3, status3 = config.resolve(match)
    assert status3 == spring_io.MATCH
    assert torch.equal(s3, springs) and n3 == 20


def test_config_topology_is_not_changed_by_callers(spring_io):
    vertices, springs, rest_lengths, spring_Y = make_case()
    original_springs, original_rest = springs.clone(), rest_lengths.clone()
    config = spring_io.ConfigSpringTopology(springs, rest_lengths, 20, vertices)

    # Changing the tensors it was built from does not reach the stored copy
    springs[0] = torch.tensor([5, 6], dtype=torch.int32)
    rest_lengths[0] = 99.0
    # Nor does changing tensors it handed out
    out_springs, out_rest, _, _ = config.resolve(legacy_checkpoint(spring_Y, 20))
    out_springs[1] = torch.tensor([7, 8], dtype=torch.int32)
    out_rest[1] = 99.0

    again, again_rest, _ = config.config_state()
    assert torch.equal(again, original_springs)
    assert torch.equal(again_rest, original_rest)
