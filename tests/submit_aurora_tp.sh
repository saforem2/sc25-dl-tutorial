#!/bin/bash --login
# SPDX-License-Identifier: MIT
#
# PBS job: run the tensor-parallel tests on Aurora (Intel XPU / PVC).
#
#   cd ~/tp-test-clone && qsub tests/submit_aurora_tp.sh
#
# Environment: the builtin `frameworks` module (PyTorch for XPU) plus a
# .venv built on top of it in the working directory -- no per-user setup
# scripts or home-directory state, so anyone with Aurora access can run
# this. Runs tests/test_parallel_dims.py (unit tests, single process) and
# tests/check_tp_equivalence.py under torchrun at TP=2 (world 2) and
# TP=2 x DP=2 (world 4).
#PBS -N vit-tp-xpu
#PBS -A datascience
#PBS -q debug
#PBS -l select=1
#PBS -l filesystems=home
#PBS -l walltime=00:30:00
# (debug queue: 1 node, <= 1 h -- use `-q prod` for production runs)
#PBS -j oe

set -x
cd "${PBS_O_WORKDIR:-$HOME/tp-test-clone}" || exit 1

# --- environment: builtin frameworks module + workdir-local venv ------------
# (a venv in $HOME would break on systems where $HOME is read-only on
# compute nodes; everything below lives under the job's working directory)
module load "${FRAMEWORKS_MODULE:-frameworks/2026.1.0}"
VENV_DIR="${TP_VENV:-$PWD/.venv}"
# (re)build the venv if it is missing or incomplete -- e.g. left
# half-created by an interrupted or concurrent job in the same workdir
if ! "$VENV_DIR/bin/python3" -c "import torch" 2>/dev/null; then
    rm -rf "$VENV_DIR"
    python3 -m venv --system-site-packages "$VENV_DIR" || exit 1
fi
source "$VENV_DIR/bin/activate"
which python3

export DEVICE=xpu
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export CCL_LOG_LEVEL=warn
unset ZE_AFFINITY_MASK

python3 - <<'PYEOF' || { echo "XPU check failed"; exit 1; }
import torch

print(f"torch {torch.__version__}")
assert torch.xpu.is_available(), "torch.xpu is not available on this node"
n = torch.xpu.device_count()
print(f"xpu devices visible: {n}")
assert n >= 4, f"need >= 4 XPU devices for the TP=2 x DP=2 run, got {n}"
PYEOF

echo "================ unit tests ================"
python3 -m unittest tests.test_parallel_dims -v || exit 1

# NOTE: launch via `python3 -m torch.distributed.run`, not the `torchrun`
# script -- the frameworks module's torchrun has a shebang pointing at
# the module's python and would bypass this venv entirely.
echo "================ equivalence: TP=2, DP=1 (world 2) ================"
TP=2 DP=1 python3 -m torch.distributed.run --standalone --nproc_per_node=2 \
    tests/check_tp_equivalence.py || exit 1

echo "================ equivalence: TP=2, DP=2 (world 4) ================"
TP=2 DP=2 python3 -m torch.distributed.run --standalone --nproc_per_node=4 \
    tests/check_tp_equivalence.py || exit 1

echo "AURORA TP TESTS: ALL PASSED"
