"""Explicit spring topology in PhysTwin checkpoints.

A checkpoint stores one learned stiffness per spring (``spring_Y``), but until
now it did not store which pair of points each spring connects. On load, the
springs are rebuilt from the config (connection radius and max neighbour
count) and ``spring_Y`` is matched to them by position. That is only correct
if the rebuilt spring list comes out in exactly the same order, and with
exactly the same members, as it did at training time.

New checkpoints additionally store the spring index pairs and rest lengths.
When a checkpoint has them, loading uses them directly, so the stiffness
always stays attached to the spring it was learned for. Checkpoints without
them (all existing releases) load exactly as before.

Two safeguards on top of that:

* When the current vertices are known, the saved rest lengths are checked
  against the distances between the vertices each saved spring connects.
  Rest lengths are those distances at build time, so a checkpoint from a
  different case with the same number of vertices fails this check instead of
  silently replacing the topology.
* ``ConfigSpringTopology`` keeps an immutable copy of the topology rebuilt
  from the config, and every checkpoint is resolved against that copy. A
  trainer that loads several checkpoints in turn therefore never resolves a
  later (for example legacy) checkpoint against springs taken from an earlier
  one.

This module only depends on torch so it can be tested without a GPU.
"""

import torch

SPRINGS_KEY = "springs"
REST_LENGTHS_KEY = "rest_lengths"

# Values returned as the ``status`` of ``resolve_spring_topology``
LEGACY = "legacy"  # checkpoint has no saved topology, use the rebuilt springs
MATCH = "match"  # saved topology is identical to the rebuilt one
SAVED = "saved"  # saved topology differs from the rebuilt one, use the saved one

# Tolerance for comparing saved rest lengths with the current geometry. Rest
# lengths are computed in float64 and stored as float32, and the vertices are
# float32, so honest mismatches are around 1e-7; a different case is off by
# the scale of the object.
REST_LENGTH_ATOL = 1e-5
REST_LENGTH_RTOL = 1e-4


def spring_topology_state(springs, rest_lengths):
    """Entries to add to a checkpoint dict so the spring topology is saved.

    Tensors are detached and moved to CPU so the checkpoint does not depend on
    the device it was trained on.
    """
    return {
        SPRINGS_KEY: springs.detach().to(device="cpu", dtype=torch.int32).clone(),
        REST_LENGTHS_KEY: rest_lengths.detach()
        .to(device="cpu", dtype=torch.float32)
        .clone(),
    }


def resolve_spring_topology(
    checkpoint, rebuilt_springs, rebuilt_rest_lengths, num_vertices, vertices=None
):
    """Decide which spring topology to use when loading ``checkpoint``.

    Args:
        checkpoint: the dict returned by ``torch.load``.
        rebuilt_springs: (N, 2) int tensor built from the config at start up.
        rebuilt_rest_lengths: (N,) float tensor built alongside it.
        num_vertices: number of vertices in the simulator (object points plus
            controller points), used to check the saved indices are in range.
        vertices: optional (num_vertices, 3) tensor of the current initial
            vertex positions. When given, the saved rest lengths must equal
            the distances between the vertices each saved spring connects,
            otherwise the checkpoint is rejected as belonging to other data.

    Returns:
        ``(springs, rest_lengths, status)``. For ``LEGACY`` and ``MATCH`` the
        rebuilt tensors are returned unchanged. For ``SAVED`` the saved tensors
        are returned on the same device and dtype as the rebuilt ones.
    """
    has_springs = SPRINGS_KEY in checkpoint
    has_rest = REST_LENGTHS_KEY in checkpoint
    if not has_springs and not has_rest:
        return rebuilt_springs, rebuilt_rest_lengths, LEGACY
    if has_springs != has_rest:
        raise ValueError(
            f"Checkpoint has only one of '{SPRINGS_KEY}' and '{REST_LENGTHS_KEY}'; "
            "both are needed to restore the spring topology"
        )

    springs = torch.as_tensor(checkpoint[SPRINGS_KEY])
    rest_lengths = torch.as_tensor(checkpoint[REST_LENGTHS_KEY])

    if springs.dim() != 2 or springs.shape[1] != 2:
        raise ValueError(
            f"Saved springs must have shape (N, 2), got {tuple(springs.shape)}"
        )
    n = springs.shape[0]
    if rest_lengths.shape != (n,):
        raise ValueError(
            f"Saved rest_lengths must have shape ({n},), got {tuple(rest_lengths.shape)}"
        )
    if "spring_Y" in checkpoint and len(checkpoint["spring_Y"]) != n:
        raise ValueError(
            f"Checkpoint has {len(checkpoint['spring_Y'])} spring_Y values "
            f"but {n} saved springs"
        )
    if n > 0 and (int(springs.min()) < 0 or int(springs.max()) >= num_vertices):
        raise ValueError(
            f"Saved springs index vertices in [{int(springs.min())}, "
            f"{int(springs.max())}] but the simulator has {num_vertices} vertices; "
            "the checkpoint does not belong to this data"
        )

    if vertices is not None:
        check_rest_lengths_match_geometry(springs, rest_lengths, vertices)

    springs = springs.to(device=rebuilt_springs.device, dtype=rebuilt_springs.dtype)
    rest_lengths = rest_lengths.to(
        device=rebuilt_rest_lengths.device, dtype=rebuilt_rest_lengths.dtype
    )

    if torch.equal(springs, rebuilt_springs) and torch.equal(
        rest_lengths, rebuilt_rest_lengths
    ):
        return rebuilt_springs, rebuilt_rest_lengths, MATCH
    return springs, rest_lengths, SAVED


def check_rest_lengths_match_geometry(springs, rest_lengths, vertices):
    """Raise ``ValueError`` if ``rest_lengths`` are not the lengths of
    ``springs`` measured on ``vertices``."""
    vertices = torch.as_tensor(vertices)
    if vertices.dim() != 2 or vertices.shape[1] != 3:
        raise ValueError(
            f"vertices must have shape (N, 3), got {tuple(vertices.shape)}"
        )
    if springs.shape[0] == 0:
        return
    v = vertices.detach().to(device="cpu", dtype=torch.float64)
    idx = springs.to(device="cpu", dtype=torch.long)
    measured = torch.linalg.norm(v[idx[:, 0]] - v[idx[:, 1]], dim=1)
    saved = rest_lengths.detach().to(device="cpu", dtype=torch.float64)
    error = (measured - saved).abs()
    allowed = REST_LENGTH_ATOL + REST_LENGTH_RTOL * saved.abs()
    bad = error > allowed
    if bool(bad.any()):
        worst = int(error.argmax())
        raise ValueError(
            f"{int(bad.sum())} of {len(saved)} saved rest lengths do not match "
            f"the current vertices (worst: spring {worst} saved "
            f"{float(saved[worst]):.6g}, measured {float(measured[worst]):.6g}); "
            "the checkpoint does not belong to this data"
        )


class ConfigSpringTopology:
    """The spring topology rebuilt from the config, kept as an immutable copy.

    Every checkpoint is resolved against this copy rather than against
    whatever topology the simulator currently holds, so loading one
    checkpoint can never change how a later one is resolved.
    """

    def __init__(self, springs, rest_lengths, num_object_springs, vertices):
        self._springs = springs.detach().clone()
        self._rest_lengths = rest_lengths.detach().clone()
        self._num_object_springs = int(num_object_springs)
        self._vertices = vertices.detach().clone()

    @property
    def num_vertices(self):
        return self._vertices.shape[0]

    @property
    def num_springs(self):
        return self._springs.shape[0]

    def config_state(self):
        """Fresh copies of the config topology:
        ``(springs, rest_lengths, num_object_springs)``."""
        return (
            self._springs.clone(),
            self._rest_lengths.clone(),
            self._num_object_springs,
        )

    def resolve(self, checkpoint):
        """Topology to use for ``checkpoint``.

        Returns ``(springs, rest_lengths, num_object_springs, status)``. The
        tensors are always fresh copies, so the caller may keep or modify them
        without touching the stored config topology.
        """
        springs, rest_lengths, status = resolve_spring_topology(
            checkpoint,
            self._springs,
            self._rest_lengths,
            self.num_vertices,
            vertices=self._vertices,
        )
        if status == SAVED:
            return (
                springs.clone(),
                rest_lengths.clone(),
                int(checkpoint["num_object_springs"]),
                status,
            )
        return (*self.config_state(), status)
