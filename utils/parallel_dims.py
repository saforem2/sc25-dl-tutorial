"""Parallel-degree bookkeeping for the DTensor (TP + FSDP2) path in train.py.

This mirrors the layout used by ezpz's ``fsdp_tp`` example: tensor parallel
("tp") is the innermost dimension (kept within a node, where interconnect
bandwidth is highest), and data parallel ("dp") runs across the remaining
ranks. The dp dimension can itself be split, torchtitan-style, into
``dp_replicate`` (weights replicated, DDP-like) and ``dp_shard`` (weights
sharded, FSDP-like) degrees, with ``dp = dp_replicate * dp_shard``. The
defaults (``dp_replicate=1``, ``dp_shard=-1``) give a single flat,
fully-sharded dp dimension.
"""

import torch
from torch.distributed.device_mesh import init_device_mesh


def resolve_dp_degrees(world_size, tp, dp_replicate=1, dp_shard=-1):
    """Resolve and validate the ``(dp_replicate, dp_shard)`` degrees.

    ``dp_shard=-1`` means "use all remaining ranks", i.e.
    ``world_size // (dp_replicate * tp)``. The product
    ``dp_replicate * dp_shard * tp`` must equal ``world_size``.

    Returns the resolved ``(dp_replicate, dp_shard)`` tuple and raises
    ``AssertionError`` with an explicit message on an invalid configuration.
    """
    assert tp >= 1, f"--tensor_parallel must be >= 1 (got {tp})"
    assert dp_replicate >= 1, f"--dp_replicate must be >= 1 (got {dp_replicate})"
    assert dp_shard == -1 or dp_shard >= 1, (
        f"--dp_shard must be >= 1, or -1 for auto (got {dp_shard})"
    )
    if dp_shard < 0:
        denom = dp_replicate * tp
        assert world_size % denom == 0, (
            f"cannot auto-resolve --dp_shard: WORLD_SIZE ({world_size}) is not "
            f"divisible by dp_replicate ({dp_replicate}) * tp ({tp}) = {denom}"
        )
        dp_shard = world_size // denom
    assert dp_replicate * dp_shard * tp == world_size, (
        f"dp_replicate ({dp_replicate}) * dp_shard ({dp_shard}) * tp ({tp}) "
        f"= {dp_replicate * dp_shard * tp} != WORLD_SIZE ({world_size})"
    )
    return dp_replicate, dp_shard


class ParallelDims:
    """Resolved parallel layout plus the DeviceMesh built from it.

    Attributes:
        tp: tensor-parallel degree.
        dp_replicate / dp_shard: the two data-parallel degrees.
        dp: total data-parallel degree (``dp_replicate * dp_shard``). This is
            the degree that data loading and loss/metric reductions must use:
            every rank in a tp group processes the *same* batch, so distinct
            samples are only spread across dp.
        mesh: the full DeviceMesh, or None when tp == dp == 1.
        tp_mesh / fsdp_mesh: sub-meshes handed to ``parallelize_module`` and
            ``fully_shard`` respectively. ``fsdp_mesh`` is 2-D
            (replicate, shard) when ``dp_replicate > 1`` so FSDP2 treats it
            as HSDP, and 1-D otherwise.
    """

    def __init__(self, world_size, tp=1, dp_replicate=1, dp_shard=-1):
        self.tp = tp
        self.dp_replicate, self.dp_shard = resolve_dp_degrees(
            world_size, tp, dp_replicate, dp_shard
        )
        self.dp = self.dp_replicate * self.dp_shard
        self.world_size = world_size
        self.mesh = None
        self.tp_mesh = None
        self.fsdp_mesh = None
        self._dp_mesh = None

    @property
    def enabled(self):
        """True when a mesh is needed (any degree > 1, or HSDP requested)."""
        return self.tp > 1 or self.dp_replicate > 1

    def build_mesh(self, device_type="cuda"):
        """Build the DeviceMesh. Requires torch.distributed to be initialized."""
        if self.mesh is not None:
            return self.mesh
        if self.dp_replicate > 1:
            self.mesh = init_device_mesh(
                device_type,
                (self.dp_replicate, self.dp_shard, self.tp),
                mesh_dim_names=("dp_replicate", "dp_shard", "tp"),
            )
            self.fsdp_mesh = self.mesh[("dp_replicate", "dp_shard")]
        else:
            self.mesh = init_device_mesh(
                device_type,
                (self.dp_shard, self.tp),
                mesh_dim_names=("dp", "tp"),
            )
            self.fsdp_mesh = self.mesh["dp"]
        self.tp_mesh = self.mesh["tp"]
        # a single flattened view of *all* data-parallel ranks, used for
        # loss/metric reductions and for the data-loader shard id
        self._dp_mesh = self.mesh.flatten(
            "dp_flat", mesh_dim_names=(
                ("dp_replicate", "dp_shard") if self.dp_replicate > 1 else ("dp",)
            )
        ) if hasattr(self.mesh, "flatten") else self.fsdp_mesh
        return self.mesh

    def _require_mesh(self):
        if self.mesh is None:
            raise RuntimeError("ParallelDims.build_mesh() must be called first")

    @property
    def dp_group(self):
        """Process group spanning all data-parallel ranks (tp peers excluded)."""
        self._require_mesh()
        return self._dp_mesh.get_group()

    @property
    def dp_rank(self):
        """This rank's index within the data-parallel dimension."""
        self._require_mesh()
        return self._dp_mesh.get_local_rank()

    @property
    def tp_rank(self):
        self._require_mesh()
        return self.tp_mesh.get_local_rank()


def validate_tp_for_vit(embed_dim, num_heads, mlp_ratio, patch_size, out_chans, tp):
    """Check that a ViT configuration can be tensor-parallelized ``tp``-way.

    The block plan shards attention by whole heads and the MLP / head along
    their feature dimensions, so every sharded dimension must divide evenly.
    Raises ``AssertionError`` naming the offending dimension.
    """
    if tp == 1:
        return
    hidden_dim = int(embed_dim * mlp_ratio)
    head_out = out_chans * patch_size * patch_size
    checks = [
        ("num_heads", num_heads),
        ("embed_dim", embed_dim),
        ("mlp hidden dim (embed_dim * mlp_ratio)", hidden_dim),
        ("head out_features (n_out_channels * patch_size**2)", head_out),
    ]
    for name, dim in checks:
        assert dim % tp == 0, (
            f"tensor_parallel={tp} does not evenly divide {name}={dim}; "
            "choose a --tensor_parallel that divides the attention heads, "
            "embed_dim, MLP hidden dim, and head output dim"
        )


def local_num_heads(num_heads, tp):
    """Attention heads owned by one rank after tp-way head sharding."""
    return num_heads // tp
