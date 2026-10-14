#!/bin/bash
#PBS -N vit-tp-xpu
#PBS -l select=1
#PBS -l walltime=00:30:00
#PBS -l filesystems=home
#PBS -q debug
#PBS -S /bin/bash
#
# Run the tensor-parallel (DTensor + FSDP2) correctness tests on Sunspot's
# Intel Data Center GPU Max (XPU) devices.
#
# Submit with your allocation and queue, e.g.:
#
#   qsub -A <project> -q <queue> tests/submit_sunspot_tp.sh
#
# (the #PBS -q above is just a default; -A has no sensible default and must
# be supplied here or added as a #PBS -A line.)
#
# Knobs (environment, pass with `qsub -v TP=2,DP=2`):
#   TP, DP          tensor/data parallel degrees for the equivalence check
#   PYTORCH_MODULE  module to load for PyTorch (default below); the script
#                   prints `module list` and the torch/XPU versions it found
#
# Scope: this validates the TP sharding + FSDP2 machinery from train.py
# (networks/tensor_parallel.py, utils/parallel_dims.py, the TP-aware
# Attention in networks/vit.py) on XPU -- forward/backward equivalence
# against the single-process model, including the oneCCL backend and an
# "xpu" DeviceMesh. train.py itself is still CUDA-targeted (it is the
# Perlmutter tutorial script); porting its training loop to XPU is separate
# from validating the parallelization code.
set -ux

cd "${PBS_O_WORKDIR:-$PWD}"

# --- software environment -------------------------------------------------
module purge 2>/dev/null || true
module use /soft/modulefiles 2>/dev/null || true
module load "${PYTORCH_MODULE:-pytorch/2.7.1}" || {
    echo "!! could not load PYTORCH_MODULE=${PYTORCH_MODULE:-pytorch/2.7.1}"
    echo "!! available pytorch modules:"
    module avail pytorch 2>&1 | tail -20
    exit 1
}
module list

which python
python -c "import torch; print('torch:', torch.__version__); \
print('xpu available:', torch.xpu.is_available()); \
print('xpu devices:', torch.xpu.device_count())"
python -c "import torch; assert torch.xpu.is_available(), 'no XPU visible'; \
assert torch.xpu.device_count() >= 2, 'need >= 2 XPU devices (tiles) for TP'"

# oneCCL is the XPU distributed backend; keep its logging quiet unless debugging
export CCL_LOG_LEVEL="${CCL_LOG_LEVEL:-warn}"
# pin each spawned rank to a distinct tile: the check assigns devices itself
# (rank % device_count), so make sure no affinity mask hides tiles from it
unset ZE_AFFINITY_MASK || true

export DEVICE=xpu

# --- tests ------------------------------------------------------------------
# 1. pure-bookkeeping unit tests (parallel degree resolution / validation)
python -m unittest tests.test_parallel_dims -v

# 2. forward/backward equivalence: tp=2, dp=1 (2 ranks on 2 tiles)
TP="${TP:-2}" DP="${DP:-1}" python tests/check_tp_equivalence.py

# 3. 2-D mesh: tp=2, dp=2 (4 ranks) -- also exercises FSDP2 grad averaging
#    over the dp mesh and the dp-group reductions train.py now uses
TP=2 DP=2 python tests/check_tp_equivalence.py

echo "SUNSPOT TP TESTS: ALL PASSED"
