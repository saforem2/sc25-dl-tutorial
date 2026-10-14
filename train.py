import sys
import os
import time
import random
import numpy as np
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
import torch.multiprocessing
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel

import logging
from utils import logging_utils

logging_utils.config_logger()
from utils.YParams import YParams
from utils import get_data_loader_distributed
from utils.loss import l2_loss, l2_loss_opt
from utils.metrics import weighted_rmse
from utils.plots import generate_images
from utils.parallel_dims import ParallelDims, validate_tp_for_vit
from networks import vit
from networks.tensor_parallel import (
    apply_fsdp2,
    mixed_precision_policy,
    parallelize_vit,
)


def train(params, args, local_rank, world_rank, world_size):
    # set device and benchmark mode
    torch.backends.cudnn.benchmark = True
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda:%d" % local_rank)

    # Data-parallel degree. With tensor parallelism, every rank in a tp
    # group processes the *same* batch and produces the same (replicated)
    # loss, so loss/metric reductions and their divisors must use the dp
    # degree/group, not the world size -- reducing over the world would
    # count each sample tp times. Without tp these are exactly the world
    # size / world group, i.e. the original behaviour.
    dp_size = getattr(args, "dp_size", world_size)
    dp_group = getattr(args, "dp_group", None)

    def all_reduce(tensor):
        if dp_group is not None:
            torch.distributed.all_reduce(tensor, group=dp_group)
        else:
            torch.distributed.all_reduce(tensor)

    # get data loader
    logging.info("rank %d, begin data loader init" % world_rank)
    train_data_loader, train_dataset, train_sampler = get_data_loader_distributed(
        params, params.train_data_path, params.distributed, train=True
    )
    val_data_loader, valid_dataset = get_data_loader_distributed(
        params, params.valid_data_path, params.distributed, train=False
    )
    logging.info("rank %d, data loader initialized" % (world_rank))

    # create model
    model = vit.ViT(params).to(device)

    # Tensor parallelism (networks/tensor_parallel.py, adapted from ezpz's
    # fsdp_tp example) replaces the DDP wrap: the model is first sharded
    # over the tp mesh with parallelize_module, then sharded over the dp
    # mesh with FSDP2 (fully_shard). FSDP2 is also available on its own via
    # --fsdp. Note this does *not* use the hand-rolled distributed layers
    # from train_mp.py -- utils.comm is never initialized here, so vit.py
    # builds its plain Attention/MLP modules, which is what the DTensor
    # plans expect.
    tp_size = getattr(args, "tensor_parallel", 1)
    use_fsdp = params.distributed and (tp_size > 1 or getattr(args, "fsdp", False))
    if use_fsdp:
        if tp_size > 1:
            assert float(params.dropout) == 0.0, (
                "tensor parallelism currently requires dropout=0.0: dropout "
                "on activations that are replicated across the tp group "
                "(pos_drop, attn.proj_drop, mlp's second drop) would use a "
                "different mask on each tp rank and silently desynchronize "
                "the group. Synchronized (tp-aware) dropout is future work."
            )
            model = parallelize_vit(model, args.parallel_dims.tp_mesh)
        model = apply_fsdp2(
            model,
            args.parallel_dims.fsdp_mesh,
            mp_policy=mixed_precision_policy(params.amp_dtype),
            reshard_after_forward=(
                getattr(args, "reshard_after_forward", "always") == "always"
            ),
        )

    if params.enable_jit:
        model = torch.compile(model)

    if params.amp_dtype == torch.float16:
        scaler = GradScaler("cuda")
    if params.distributed and not args.noddp and not use_fsdp:
        if args.disable_broadcast_buffers:
            model = DistributedDataParallel(
                model,
                device_ids=[local_rank],
                bucket_cap_mb=args.bucket_cap_mb,
                broadcast_buffers=False,
                gradient_as_bucket_view=True,
            )
        else:
            model = DistributedDataParallel(
                model, device_ids=[local_rank], bucket_cap_mb=args.bucket_cap_mb
            )

    if params.enable_fused:
        optimizer = optim.Adam(
            model.parameters(), lr=params.lr, fused=True, betas=(0.9, 0.95)
        )
    else:
        optimizer = optim.Adam(model.parameters(), lr=params.lr, betas=(0.9, 0.95))

    if world_rank == 0:
        logging.info(model)

    iters = 0
    startEpoch = 0

    if params.lr_schedule == "cosine":
        if params.warmup > 0:
            lr_scale = lambda x: min(
                (x + 1) / params.warmup,
                0.5 * (1 + np.cos(np.pi * x / params.num_iters)),
            )
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_scale)
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=params.num_iters
            )
    else:
        scheduler = None

    # select loss function
    if params.enable_jit:
        loss_func = l2_loss_opt
    else:
        loss_func = l2_loss

    if world_rank == 0:
        logging.info("Starting Training Loop...")

    # Log initial loss on train and validation to tensorboard
    with torch.no_grad():
        inp, tar = map(lambda x: x.to(device), next(iter(train_data_loader)))
        gen = model(inp)
        tr_loss = loss_func(gen, tar)
        inp, tar = map(lambda x: x.to(device), next(iter(val_data_loader)))
        gen = model(inp)
        val_loss = loss_func(gen, tar)
        val_rmse = weighted_rmse(gen, tar)
        if params.distributed:
            all_reduce(tr_loss)
            all_reduce(val_loss)
            all_reduce(val_rmse)
        if world_rank == 0:
            args.tboard_writer.add_scalar("Loss/train", tr_loss.item() / dp_size, 0)
            args.tboard_writer.add_scalar("Loss/valid", val_loss.item() / dp_size, 0)
            args.tboard_writer.add_scalar(
                "RMSE(u10m)/valid", val_rmse.cpu().numpy()[0] / dp_size, 0
            )

    params.num_epochs = params.num_iters // len(train_data_loader)
    iters = 0
    t1 = time.time()
    for epoch in range(startEpoch, startEpoch + params.num_epochs):
        torch.cuda.synchronize()  # device sync to ensure accurate epoch timings
        if params.distributed and (train_sampler is not None):
            train_sampler.set_epoch(epoch)
        start = time.time()
        tr_loss = []
        tr_time = 0.0
        dat_time = 0.0
        log_time = 0.0

        model.train()
        step_count = 0
        for i, data in enumerate(train_data_loader, 0):
            if world_rank == 0:
                if epoch == 3 and i == 0:
                    torch.cuda.profiler.start()
                if epoch == 3 and i == len(train_data_loader) - 1:
                    torch.cuda.profiler.stop()

            torch.cuda.nvtx.range_push(f"step {i}")
            iters += 1
            dat_start = time.time()
            torch.cuda.nvtx.range_push(f"data copy in {i}")

            inp, tar = map(lambda x: x.to(device), data)
            torch.cuda.nvtx.range_pop()  # copy in

            tr_start = time.time()
            b_size = inp.size(0)

            optimizer.zero_grad()

            torch.cuda.nvtx.range_push(f"forward")
            with autocast("cuda", enabled=params.amp_enabled, dtype=params.amp_dtype):
                gen = model(inp)
                loss = loss_func(gen, tar)
            torch.cuda.nvtx.range_pop()  # forward

            if params.amp_dtype == torch.float16:
                scaler.scale(loss).backward()
                torch.cuda.nvtx.range_push(f"optimizer")
                scaler.step(optimizer)
                torch.cuda.nvtx.range_pop()  # optimizer
                scaler.update()
            else:
                loss.backward()
                torch.cuda.nvtx.range_push(f"optimizer")
                optimizer.step()
                torch.cuda.nvtx.range_pop()  # optimizer

            if params.distributed:
                all_reduce(loss)
            tr_loss.append(loss.item() / dp_size)

            torch.cuda.nvtx.range_pop()  # step
            # lr step
            scheduler.step()

            tr_end = time.time()
            tr_time += tr_end - tr_start
            dat_time += tr_start - dat_start
            step_count += 1

        torch.cuda.synchronize()  # device sync to ensure accurate epoch timings
        end = time.time()

        if world_rank == 0:
            iters_per_sec = step_count / (end - start)
            samples_per_sec = params["global_batch_size"] * iters_per_sec
            logging.info(
                "Time taken for epoch %i is %f sec, avg %f samples/sec",
                epoch + 1,
                end - start,
                samples_per_sec,
            )
            logging.info("  Avg train loss=%f" % np.mean(tr_loss))
            args.tboard_writer.add_scalar("Loss/train", np.mean(tr_loss), iters)
            args.tboard_writer.add_scalar(
                "Learning Rate", optimizer.param_groups[0]["lr"], iters
            )
            args.tboard_writer.add_scalar("Avg iters per sec", iters_per_sec, iters)
            args.tboard_writer.add_scalar("Avg samples per sec", samples_per_sec, iters)
            fig = generate_images([inp, tar, gen])
            args.tboard_writer.add_figure("Visualization, t2m", fig, iters, close=True)

        val_start = time.time()
        val_loss = torch.zeros(1, device=device)
        val_rmse = torch.zeros(
            (params.n_out_channels), dtype=torch.float32, device=device
        )
        valid_steps = 0
        model.eval()

        with torch.inference_mode():
            with torch.no_grad():
                for i, data in enumerate(val_data_loader, 0):
                    with autocast(
                        "cuda", enabled=params.amp_enabled, dtype=params.amp_dtype
                    ):
                        inp, tar = map(lambda x: x.to(device), data)
                        gen = model(inp)
                        val_loss += loss_func(gen, tar)
                        val_rmse += weighted_rmse(gen, tar)
                    valid_steps += 1

                if params.distributed:
                    all_reduce(val_loss)
                    val_loss /= dp_size
                    all_reduce(val_rmse)
                    val_rmse /= dp_size

        val_rmse /= valid_steps  # Avg validation rmse
        val_loss /= valid_steps
        val_end = time.time()
        if world_rank == 0:
            logging.info("  Avg val loss={}".format(val_loss.item()))
            logging.info("  Total validation time: {} sec".format(val_end - val_start))
            args.tboard_writer.add_scalar("Loss/valid", val_loss, iters)
            args.tboard_writer.add_scalar(
                "RMSE(u10m)/valid", val_rmse.cpu().numpy()[0], iters
            )
            args.tboard_writer.flush()

    t2 = time.time()
    tottime = t2 - t1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run_num",
        default="00",
        type=str,
        help="tag for indexing the current experiment",
    )
    parser.add_argument(
        "--yaml_config",
        default="./config/ViT.yaml",
        type=str,
        help="path to yaml file containing training configs",
    )
    parser.add_argument(
        "--config", default="base", type=str, help="name of desired config in yaml file"
    )
    parser.add_argument(
        "--amp_mode",
        default="none",
        type=str,
        choices=["none", "fp16", "bf16"],
        help="select automatic mixed precision mode",
    )
    parser.add_argument(
        "--enable_fused", action="store_true", help="enable fused Adam optimizer"
    )
    parser.add_argument(
        "--enable_jit", action="store_true", help="enable JIT compilation"
    )
    parser.add_argument(
        "--local_batch_size",
        default=None,
        type=int,
        help="local batchsize (manually override global_batch_size config setting)",
    )
    parser.add_argument(
        "--num_iters", default=None, type=int, help="number of iters to run"
    )
    parser.add_argument(
        "--num_data_workers",
        default=None,
        type=int,
        help="number of data workers for data loader",
    )
    parser.add_argument(
        "--data_loader_config",
        default=None,
        type=str,
        choices=["pytorch", "dali"],
        help="dataloader configuration. choices: 'pytorch', 'dali'",
    )
    parser.add_argument(
        "--bucket_cap_mb", default=25, type=int, help="max message bucket size in mb"
    )
    parser.add_argument(
        "--disable_broadcast_buffers",
        action="store_true",
        help="disable syncing broadcasting buffers",
    )
    parser.add_argument(
        "--noddp", action="store_true", help="disable DDP communication"
    )

    # tensor parallelism / FSDP2 arguments (see networks/tensor_parallel.py
    # and utils/parallel_dims.py). --tensor_parallel > 1 implies --fsdp.
    parser.add_argument(
        "--tensor_parallel",
        "--tp",
        default=1,
        type=int,
        help="number of GPUs for tensor parallelism (DTensor, FSDP2-based)",
    )
    parser.add_argument(
        "--dp_replicate",
        default=1,
        type=int,
        help="data-parallel replication degree for HSDP "
        "(dp = dp_replicate * dp_shard); FSDP2 path only",
    )
    parser.add_argument(
        "--dp_shard",
        default=-1,
        type=int,
        help="data-parallel sharding degree for HSDP; -1 (default) uses "
        "all remaining ranks. FSDP2 path only",
    )
    parser.add_argument(
        "--fsdp",
        action="store_true",
        help="use FSDP2 (fully_shard) over the data-parallel mesh instead "
        "of DDP; implied by --tensor_parallel > 1",
    )
    parser.add_argument(
        "--reshard_after_forward",
        default="always",
        type=str,
        choices=["always", "never"],
        help="FSDP2 reshard policy: 'always' (ZeRO-3 style, default) "
        "re-shards parameters after forward; 'never' (ZeRO-2 style) keeps "
        "them gathered. FSDP2 path only",
    )
    parser.add_argument(
        "--seed",
        default=None,
        type=int,
        help="seed for model initialization; defaults to 0 when tensor "
        "parallelism is enabled, since all tp ranks must start from "
        "identical weights (DDP's rank-0 broadcast no longer does this)",
    )
    args = parser.parse_args()

    run_num = args.run_num

    params = YParams(os.path.abspath(args.yaml_config), args.config)

    # Update config with modified args
    # set up amp
    if args.amp_mode != "none":
        params.update({"amp_mode": args.amp_mode})
    amp_dtype = torch.float32

    if params.amp_mode == "fp16":
        amp_dtype = torch.float16
    elif params.amp_mode == "bf16":
        amp_dtype = torch.bfloat16

    params.update(
        {"amp_enabled": amp_dtype is not torch.float32, "amp_dtype": amp_dtype}
    )

    if args.enable_fused:
        params.update({"enable_fused": args.enable_fused})

    if args.enable_jit:
        params.update({"enable_jit": args.enable_jit})

    if args.data_loader_config:
        params.update({"data_loader_config": args.data_loader_config})

    if args.num_iters:
        params.update({"num_iters": args.num_iters})

    if args.num_data_workers:
        params.update({"num_data_workers": args.num_data_workers})

    params.distributed = False
    if "WORLD_SIZE" in os.environ:
        params.distributed = int(os.environ["WORLD_SIZE"]) > 1
        world_size = int(os.environ["WORLD_SIZE"])
    else:
        world_size = 1

    world_rank = 0
    local_rank = 0
    if params.distributed:
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        world_rank = torch.distributed.get_rank()
        local_rank = int(os.environ["LOCAL_RANK"])

    # Work out the parallel layout. Ranks are arranged as dp x tp, with tp
    # innermost (contiguous ranks, normally within a node). The dp degree
    # -- *not* the world size -- determines how the global batch and the
    # dataset are split: every rank in a tp group must see the same batch.
    use_fsdp_path = params.distributed and (args.tensor_parallel > 1 or args.fsdp)
    if args.tensor_parallel > 1 and not params.distributed:
        parser.error(
            "--tensor_parallel > 1 requires a distributed launch (WORLD_SIZE > 1)"
        )
    if (args.dp_replicate != 1 or args.dp_shard != -1) and not (
        args.tensor_parallel > 1 or args.fsdp
    ):
        logging.warning(
            "--dp_replicate/--dp_shard only apply to the FSDP2 path "
            "(--tensor_parallel > 1 or --fsdp); ignoring them"
        )
        args.dp_replicate, args.dp_shard = 1, -1
    if args.seed is None and args.tensor_parallel > 1:
        args.seed = 0
    parallel_dims = ParallelDims(
        world_size,
        tp=args.tensor_parallel,
        dp_replicate=args.dp_replicate,
        dp_shard=args.dp_shard,
    )
    args.parallel_dims = parallel_dims
    args.dp_size = parallel_dims.dp
    args.dp_group = None
    args.dp_rank = world_rank
    if args.tensor_parallel > 1:
        validate_tp_for_vit(
            embed_dim=params.embed_dim,
            num_heads=params.num_heads,
            mlp_ratio=4,  # hard-coded in networks.vit.ViT()
            patch_size=params.patch_size,
            out_chans=params.n_out_channels,
            tp=args.tensor_parallel,
        )
    if use_fsdp_path:
        parallel_dims.build_mesh(device_type="cuda")
        args.dp_group = parallel_dims.dp_group
        args.dp_rank = parallel_dims.dp_rank
        if world_rank == 0:
            logging.info(
                "Parallel layout: tp=%d, dp_replicate=%d, dp_shard=%d (dp=%d), "
                "world_size=%d",
                parallel_dims.tp,
                parallel_dims.dp_replicate,
                parallel_dims.dp_shard,
                parallel_dims.dp,
                world_size,
            )
    dp_size = parallel_dims.dp

    if args.local_batch_size:
        # Manually override batch size
        params.local_batch_size = args.local_batch_size
        params.update({"global_batch_size": dp_size * args.local_batch_size})
    else:
        # Compute local batch size based on the number of *data-parallel*
        # ranks (tp ranks share the same samples)
        assert params.global_batch_size % dp_size == 0, (
            f"global_batch_size ({params.global_batch_size}) must be divisible "
            f"by the data-parallel degree ({dp_size})"
        )
        params.local_batch_size = params.global_batch_size // dp_size

    # for the data loaders, set the actual number of data shards and id:
    # shards are indexed by dp rank, so tp peers load identical batches
    params.data_num_shards = dp_size
    params.data_shard_id = args.dp_rank

    # Set up directory
    baseDir = params.expdir
    expDir = os.path.join(
        baseDir, args.config + "/%dGPU/" % (world_size) + str(run_num) + "/"
    )
    if world_rank == 0:
        if not os.path.isdir(expDir):
            os.makedirs(expDir)
        logging_utils.log_to_file(
            logger_name=None, log_filename=os.path.join(expDir, "out.log")
        )
        params.log()
        args.tboard_writer = SummaryWriter(log_dir=os.path.join(expDir, "logs/"))

    params.experiment_dir = os.path.abspath(expDir)

    if args.seed is not None:
        # all ranks use the same seed: under tensor parallelism every tp
        # rank must construct identical (pre-shard) weights, and nothing
        # broadcasts them the way DDP does
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    train(params, args, local_rank, world_rank, world_size)

    if params.distributed:
        torch.distributed.barrier()
    logging.info("DONE ---- rank %d" % world_rank)
