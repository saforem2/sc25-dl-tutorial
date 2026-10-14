#!/bin/bash --login
# SPDX-License-Identifier: MIT
#
# PBS job: run the tensor-parallel tests on Sunspot (Intel XPU / PVC).
#
#   cd ~/tp-test-clone && qsub tests/submit_sunspot_tp.sh
#
# Uses the frameworks RC environment (~/rc_env.sh: modules + conda env with
# PyTorch built for XPU) plus the rc4-ezpz venv layered on top of it.
# Runs tests/test_parallel_dims.py (unit tests, single process) and
# tests/check_tp_equivalence.py under torchrun at TP=2 (world 2) and
# TP=2 x DP=2 (world 4).
#PBS -N vit-tp-xpu
#PBS -A datascience
#PBS -q workq
#PBS -l select=1
#PBS -l filesystems=home
#PBS -l walltime=00:30:00
#PBS -j oe

set -x
cd "${PBS_O_WORKDIR:-$HOME/tp-test-clone}" || exit 1

# --- environment (frameworks RC: oneAPI + PyTorch for XPU) -------------------
source "$HOME/rc_env.sh"
source "$HOME/venvs/rc4-ezpz/bin/activate"
which python3

export DEVICE=xpu
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

echo "================ equivalence: TP=2, DP=1 (world 2) ================"
TP=2 DP=1 torchrun --standalone --nproc_per_node=2 \
    tests/check_tp_equivalence.py || exit 1

echo "================ equivalence: TP=2, DP=2 (world 4) ================"
TP=2 DP=2 torchrun --standalone --nproc_per_node=4 \
    tests/check_tp_equivalence.py || exit 1

echo "SUNSPOT TP TESTS: ALL PASSED"
