# Tensor parallelism in `train.py` (DTensor + FSDP2)

`train.py` can shard the ViT across devices with **tensor parallelism**
(DTensor `parallelize_module`) and shard parameters/gradients/optimizer
state across data-parallel ranks with **FSDP2** (`fully_shard`) -- the
same combination as the `fsdp_tp` example in
[ezpz](https://github.com/saforem2/ezpz), expressed declaratively: a
per-module sharding plan instead of hand-written parallel modules.
Setting `--tensor_parallel=1` without `--fsdp` keeps the original
single-device / DDP behavior.

This page is the run book: both launcher variants, expected output, and
example configs. For the concepts, see the "Tensor parallelism" section
of the [README](README.md).

## Parallel layout

Ranks are arranged as a 2-D mesh, tp **innermost** (contiguous ranks,
normally within a node):

```
world = dp_replicate x dp_shard x tp        dp = dp_replicate * dp_shard
```

| flag | default | meaning |
|---|---|---|
| `--tensor_parallel` / `--tp` | 1 | tensor-parallel degree (attention heads, MLP, head sharded) |
| `--dp_replicate` | 1 | dp degree over which weights are replicated (DDP-like) |
| `--dp_shard` | -1 (auto) | dp degree over which weights are FSDP-sharded |
| `--fsdp` | off | force the FSDP2 path even at tp=1 |
| `--device` | auto | `cuda` / `xpu` / `cpu` (auto = cuda, else xpu, else cpu) |
| `--ezpz` | off | delegate setup + device selection to ezpz (variant B) |

Notes:

- Data loading, loss, and RMSE reductions are all taken over the **dp**
  degree only -- ranks in a tp group see the same batch.
- `dropout` must be `0.0` with tp > 1 (as in the `mp*` configs): sharded
  dropout would need a tp-aware RNG offset. Checked up front.
- tp must evenly divide `num_heads`, `embed_dim`, the MLP hidden dim,
  and the head's output dim; checked up front with a named assert.
- Sharding plan (per transformer block): q/k/v column-parallel, attn
  proj row-parallel, fc1 column-parallel, fc2 row-parallel; the root
  head is column-parallel with a replicated output. Patch embedding,
  norms, and position embedding stay replicated. There is no sequence
  parallelism in this version.

## Environment

**Perlmutter** (the tutorial's home): the repo's `submit_pm*.sh`
scripts and the `nersc/pytorch` container, as in the README. No extra
packages are needed for variant A.

**ALCF Aurora / Sunspot** (Intel XPU, PVC): load the builtin frameworks
module and layer a workdir-local venv over it -- the PBS scripts below
do this automatically:

```bash
module load frameworks/2026.1.0            # torch 2.13 XPU build
python3 -m venv --system-site-packages .venv && source .venv/bin/activate
pip install tensorboard ruamel.yaml         # the two deps the module lacks
```

XPU specifics: the process-group backend is `xccl` (PyTorch >= 2.5 has
native XPU support; IPEX is *not* required). Launch with
`python3 -m torch.distributed.run`, not the module's `torchrun` script
-- its shebang points at the module python and bypasses your venv.

Variant B additionally needs ezpz, installed **from git** (the PyPI
package named `ezpz` is an unrelated image tool):

```bash
pip install "git+https://github.com/saforem2/ezpz"
```

## Variant A: standard (built-in setup)

`train.py` reads `RANK` / `WORLD_SIZE` / `LOCAL_RANK` from the
environment and picks the backend from the device type itself.

Single node (any machine), 4-way TP smoke run on synthetic data:

```bash
python3 tests/make_synthetic_era5.py --out ./data/synthetic
python3 -m torch.distributed.run --standalone --nproc_per_node=4 \
    train.py --yaml_config=./config/ViT.yaml --config=mp_smoke_dt \
             --tensor_parallel=4 --device=xpu --run_num=tp4   # --device=cuda on NVIDIA
```

Aurora, as a PBS job (tests + both training variants have scripts):

```bash
qsub tests/submit_aurora_tp.sh               # correctness tests
qsub tests/submit_aurora_train_tp.sh         # this training run
```

Perlmutter, 16 GPUs (4 nodes), tp=4 x dp=4 on the real dataset:

```bash
sbatch --nodes 4 submit_pm_dp.sh --config=mp_short \
       --tensor_parallel=4 --run_num=tp4
```

## Variant B: ezpz-based

`--ezpz` replaces the built-in plumbing with
`ezpz.distributed.setup_torch(tensor_parallel_size=...)` -- ezpz
discovers rank/world (MPI or env), selects and sets the device, and
initializes the process group -- and `ezpz launch` builds the
scheduler-aware `mpiexec` command (hostfile, CPU binding) for you:

```bash
ezpz launch --nproc 4 \
    python3 train.py --yaml_config=./config/ViT.yaml --config=mp_smoke_dt \
             --tensor_parallel=4 --ezpz --run_num=tp4ezpz
# Aurora, as a PBS job:
qsub tests/submit_aurora_train_tp_ezpz.sh
```

Two implementation details, if you read the code:

- `setup_torch` is called with `seed=None`: its seeding is rank-aware,
  which would desynchronize weight init across tp ranks. `train.py`
  seeds identically on all ranks itself (and forces `seed=0` at tp>1).
- With `--ezpz` the DeviceMesh is built via
  `ezpz.init_device_mesh_safe`: ezpz creates a *device-bound* default
  process group, under which torch's mesh construction takes the
  `split_group` path -- unsupported by xccl on XPU. The safe helper
  falls back to `new_group`. (ezpz's own `fsdp_tp` does the same.)

## Expected output

Correctness tests (job `submit_aurora_tp.sh`):

```
Ran 12 tests in ... OK
OK: tp=2, dp=1, world=2
OK: tp=2, dp=2, world=4
AURORA TP TESTS: ALL PASSED
```

The equivalence checks compare forward outputs, parameters, and
gradients of the sharded model against the single-process model (built
on rank 0 and broadcast), on the job's device.

Training smoke run (either variant -- they are numerically identical;
same seed, data, and init):

```
Device: xpu (process-group backend: xccl)
Parallel layout: tp=4, dp_replicate=1, dp_shard=1 (dp=1), world_size=4
Starting Training Loop...
Time taken for epoch 1 is 7.48 sec, avg 8.55 samples/sec
  Avg train loss=1.108552
  Avg val loss=1.017984
Time taken for epoch 2 is 3.05 sec, avg 20.96 samples/sec
  Avg train loss=1.008547
  Avg val loss=1.004148
DONE ---- rank 0
```

(Timings vary by machine/load; losses should reproduce exactly for
`mp_smoke_dt`, which uses the deterministic synthetic dataset and
`seed=0`.)

## Example configs to try

| config | purpose | suggested launch |
|---|---|---|
| `mp_smoke_dt` | 20-iter smoke run, `embed_dim=1024`, depth 12, synthetic ERA5 subset | tp=4, 1 node (above) |
| `mp_short` | short run on the real ERA5 subset | `sbatch --nodes 4 submit_pm_dp.sh --config=mp_short --tensor_parallel=4` (tp=4 x dp=4) |
| `mp` | full model-parallel config from the tutorial | `--tensor_parallel=4`, scale nodes for dp |
| any of the above + HSDP | replicate x shard data parallel | add `--dp_replicate=2 --dp_shard=2` with world = 2 x 2 x tp (e.g. tp=2 on 8 ranks) |

Unit tests for the layout bookkeeping run anywhere, no torch needed:

```bash
python3 -m unittest tests.test_parallel_dims
```

## Troubleshooting

- `AssertionError: tensor_parallel=... does not evenly divide ...` --
  pick a tp dividing heads/embed/MLP/head-out (checked by
  `validate_tp_for_vit` before any distributed work starts).
- `RuntimeError: No backend for the parent process group ... split` --
  you built a DeviceMesh over a device-bound PG outside the ezpz-safe
  helpers; use `--ezpz` (which handles it) or the standard launcher.
- Hangs at FSDP2's first all-gather on XPU usually mean ranks bound to
  different devices; both variants set the per-rank device *before*
  `init_process_group` -- don't move that code.
- `KeyError: 'SLURM_PROCID'` in logging under PBS/torchrun is fixed:
  the rank log filter accepts `SLURM_PROCID` / `RANK` / `PMI_RANK`.

## Files

- `train.py` -- flags, setup (both variants), training loop
- `networks/tensor_parallel.py` -- sharding plan, `parallelize_vit`,
  FSDP2 application
- `utils/parallel_dims.py` -- degree resolution + DeviceMesh bookkeeping
- `tests/test_parallel_dims.py`, `tests/check_tp_equivalence.py` --
  correctness
- `tests/submit_aurora_*.sh` -- Aurora PBS wrappers (tests, train, ezpz)
- `tests/make_synthetic_era5.py` -- deterministic synthetic ERA5 subset
