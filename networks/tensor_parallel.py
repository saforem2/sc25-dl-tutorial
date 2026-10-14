"""DTensor tensor parallelism (+ FSDP2) for the ViT in networks/vit.py.

Adapted from the pattern in ezpz's ``fsdp_tp`` example (itself modified from
PyTorch's TP tutorial): each transformer block is sharded with
``parallelize_module`` over the tp sub-mesh, and the sharded blocks are then
wrapped with FSDP2 (``fully_shard``) over the dp sub-mesh, giving 2-D
TP + FSDP training.

The plan follows the usual Megatron pairing. For each ``Block``:

- ``attn.q`` / ``attn.k`` / ``attn.v``: column-wise. Each rank owns a
  contiguous subset of the attention heads (``tp`` must divide
  ``num_heads``), so attention itself needs no communication.
- ``attn.proj``: row-wise. Consumes the head-sharded attention output and
  all-reduces the partial sums back to a replicated ``embed_dim`` tensor.
- ``mlp.fc1``: column-wise (hidden dim sharded), ``mlp.fc2``: row-wise.

Everything else stays replicated on purpose:

- ``patch_embed.proj`` is a ``Conv2d`` -- the stock TP styles only target
  ``nn.Linear`` / ``nn.Embedding`` -- and it is small.
- ``norm1`` / ``norm2`` / the final ``norm`` are plain LayerNorms over the
  replicated ``embed_dim`` (no Sequence Parallel in this version).
- ``pos_embed`` is a bare Parameter, not a module, so it cannot appear in a
  parallelize plan.

The one exception at the root is ``head``: it is column-wise sharded with a
replicated output layout, so the model output is the full forecast image on
every rank and the existing L2 loss / RMSE / plotting code in train.py works
unchanged.
"""

import torch
from torch.distributed._tensor import Replicate
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    RowwiseParallel,
    parallelize_module,
)


def block_tp_plan():
    """Tensor-parallel plan for one ``networks.vit.Block``."""
    return {
        "attn.q": ColwiseParallel(),
        "attn.k": ColwiseParallel(),
        "attn.v": ColwiseParallel(),
        "attn.proj": RowwiseParallel(),
        "mlp.fc1": ColwiseParallel(),
        "mlp.fc2": RowwiseParallel(),
    }


def root_tp_plan():
    """Tensor-parallel plan for the root ``VisionTransformer`` module."""
    return {
        # shard the (large) head weight, but hand every rank the full
        # output so the loss sees a plain, replicated tensor
        "head": ColwiseParallel(output_layouts=Replicate()),
    }


def parallelize_vit(model, tp_mesh):
    """Apply tensor parallelism to a ``VisionTransformer`` in place."""
    for block in model.blocks:
        parallelize_module(block, tp_mesh, block_tp_plan())
    parallelize_module(model, tp_mesh, root_tp_plan())
    return model


def apply_fsdp2(model, fsdp_mesh, mp_policy=None, reshard_after_forward=True):
    """Wrap each block, then the root model, with FSDP2 ``fully_shard``.

    Must be called *after* :func:`parallelize_vit` (when tp > 1) so the
    TP-sharded DTensor parameters are sharded again over the dp mesh, and
    *before* the optimizer is created, so the optimizer sees the sharded
    parameters. ``reshard_after_forward=True`` is the ZeRO-3 style policy
    (parameters are re-sharded after the forward pass); ``False`` keeps them
    gathered (ZeRO-2 style, more memory, fewer collectives).
    """
    kwargs = {"mesh": fsdp_mesh, "reshard_after_forward": reshard_after_forward}
    if mp_policy is not None:
        kwargs["mp_policy"] = mp_policy
    for block in model.blocks:
        fully_shard(block, **kwargs)
    fully_shard(model, **kwargs)
    return model


def mixed_precision_policy(amp_dtype):
    """FSDP2 mixed-precision policy matching train.py's AMP settings.

    Returns None for fp32 (no policy needed). Reductions are kept in fp32,
    matching the behaviour the GradScaler/autocast path approximates.
    """
    if amp_dtype is None or amp_dtype == torch.float32:
        return None
    return MixedPrecisionPolicy(param_dtype=amp_dtype, reduce_dtype=torch.float32)
