"""Forward/backward equivalence check for the DTensor TP (+FSDP2) ViT.

Builds a small ViT, runs it single-process, then rebuilds the *same*
weights sharded with ``parallelize_vit`` + ``apply_fsdp2`` (the exact
helpers train.py uses) and checks that outputs and gradients match.

Runs on CPU, CUDA, or Intel XPU (DEVICE=cpu|cuda|xpu):

    python tests/check_tp_equivalence.py            # tp=2, dp=1 (2 procs)
    TP=2 DP=2 python tests/check_tp_equivalence.py  # 2-D mesh (4 procs)
    DEVICE=xpu TP=2 python tests/check_tp_equivalence.py   # Aurora / PVC

On XPU the distributed backend is oneCCL ("ccl") and the device mesh is
built with device_type="xpu" -- the same combination train.py's FSDP2 path
needs off-NVIDIA. See tests/submit_aurora_tp.sh for the qsub wrapper.

Exits non-zero (via the spawned workers) if any check fails.
"""

import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

# make the repo root importable no matter how this file is launched
# (torchrun/mpiexec put this file's directory, not the cwd, on sys.path)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from networks import vit
from networks.tensor_parallel import apply_fsdp2, parallelize_vit
from utils.parallel_dims import ParallelDims

ATOL = 1e-5
RTOL = 1e-4


def make_params():
    return SimpleNamespace(
        img_size=(32, 32),
        n_in_channels=3,
        n_out_channels=3,
        patch_size=8,
        embed_dim=32,
        depth=2,
        num_heads=4,
        dropout=0.0,
    )


def make_input():
    g = torch.Generator().manual_seed(123)
    return torch.randn(2, 3, 32, 32, generator=g)


def full(t):
    """DTensor -> plain tensor (identity for plain tensors)."""
    return t.full_tensor() if hasattr(t, "full_tensor") else t


def default_backend(device_type):
    if device_type == "cuda":
        return "nccl"
    if device_type == "xpu":
        # native XPU builds register oneCCL as "xccl"; older IPEX-based
        # builds called it "ccl"
        if hasattr(dist, "is_xccl_available") and dist.is_xccl_available():
            return "xccl"
        return "ccl"
    return "gloo"


BACKENDS = {dt: default_backend(dt) for dt in ("cpu", "cuda", "xpu")}


def worker(rank, world_size, tp, init_method):
    device_type = os.environ.get("DEVICE", "cpu")
    backend = BACKENDS[device_type]
    if device_type == "xpu":
        # older Intel PyTorch builds kept oneCCL support in IPEX; PyTorch
        # >= 2.5 (incl. the Aurora frameworks builds) has native XPU +
        # oneCCL support, so a missing IPEX is fine
        try:
            import intel_extension_for_pytorch  # noqa: F401
        except ImportError:
            pass

    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    dist.init_process_group(backend, init_method=init_method,
                            rank=rank, world_size=world_size)
    try:
        if device_type == "cpu":
            device = torch.device("cpu")
        else:
            set_device = getattr(torch, device_type).set_device
            device_count = getattr(torch, device_type).device_count()
            set_device(rank % device_count)
            device = torch.device(device_type, rank % device_count)

        params = make_params()
        x = make_input().to(device)

        # ---- reference: same seed + weights, no parallelism ----
        torch.manual_seed(0)
        ref = vit.ViT(params).to(device)
        ref.eval()
        ref_out = ref(x)
        ref_loss = ref_out.pow(2).mean()
        ref_loss.backward()
        ref_params = {n: p.detach().clone() for n, p in ref.named_parameters()}
        ref_grads = {
            n: p.grad.detach().clone()
            for n, p in ref.named_parameters()
            if p.grad is not None
        }

        # ---- parallel: identical weights, TP (+FSDP2) sharded ----
        torch.manual_seed(0)
        model = vit.ViT(params).to(device)
        model.load_state_dict(ref.state_dict())
        model.eval()

        dims = ParallelDims(world_size, tp=tp)
        assert dims.dp * dims.tp == world_size
        dims.build_mesh(device_type=device_type)
        if tp > 1:
            model = parallelize_vit(model, dims.tp_mesh)
        model = apply_fsdp2(model, dims.fsdp_mesh)

        out = model(x)
        assert torch.allclose(out, ref_out, atol=ATOL, rtol=RTOL), (
            f"[rank {rank}] forward mismatch: "
            f"max err {(out - ref_out).abs().max().item():.3e}"
        )

        loss = out.pow(2).mean()
        loss.backward()
        assert torch.allclose(loss, ref_loss, atol=ATOL, rtol=RTOL)

        for name, p in model.named_parameters():
            # parameter values (reconstructed across all shards) must match
            assert torch.allclose(full(p.data), ref_params[name],
                                  atol=ATOL, rtol=RTOL), (
                f"[rank {rank}] param {name} mismatch"
            )
            if name in ref_grads:
                assert p.grad is not None, f"[rank {rank}] no grad for {name}"
                # NOTE: with dp > 1 FSDP averages grads over the dp mesh,
                # and every dp replica saw the same input here, so the
                # reconstructed grad still equals the reference grad.
                assert torch.allclose(full(p.grad), ref_grads[name],
                                      atol=ATOL, rtol=RTOL), (
                    f"[rank {rank}] grad {name} mismatch: max err "
                    f"{(full(p.grad) - ref_grads[name]).abs().max().item():.3e}"
                )

        # local shard of a block weight: TP colwise-shards dim 0 by tp and
        # FSDP2 shards dim 0 over the dp mesh as well, so the local piece is
        # 1/(tp * dp) of the full rows (dp counts only the sharding degree
        # here because dp_replicate=1 in this test)
        if tp > 1:
            q = model.blocks[0].attn.q.weight
            local = q.to_local() if hasattr(q, "to_local") else q
            expect = ref_params["blocks.0.attn.q.weight"].shape[0] // (tp * dims.dp)
            assert local.shape[0] == expect, (
                f"[rank {rank}] q.weight local rows {local.shape[0]} != {expect}"
            )

        if rank == 0:
            print(
                f"OK: tp={tp}, dp={dims.dp}, world={world_size} -- forward, "
                "backward, params and grads all match the single-process model"
            )
    finally:
        dist.destroy_process_group()


def main():
    tp = int(os.environ.get("TP", "2"))
    dp = int(os.environ.get("DP", "1"))
    world_size = tp * dp
    if "LOCAL_RANK" in os.environ and "MASTER_ADDR" in os.environ:
        # launched by torchrun / mpiexec: this process already IS one rank.
        # (mp.spawn here would double every rank and deadlock the group.)
        worker(int(os.environ["RANK"]), world_size, tp, "env://")
    else:
        # standalone: spawn all ranks from this process, rendezvous via file
        with tempfile.TemporaryDirectory() as tmpdir:
            mp.spawn(
                worker,
                args=(world_size, tp,
                      f"file://{os.path.join(tmpdir, 'store')}"),
                nprocs=world_size,
                join=True,
            )


if __name__ == "__main__":
    main()
