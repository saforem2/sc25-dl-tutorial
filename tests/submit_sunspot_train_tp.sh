#!/bin/bash --login
# SPDX-License-Identifier: MIT
#
# PBS job: smoke-train train.py with DTensor tensor parallelism + FSDP2 on
# Sunspot (Intel XPU / PVC), config mp_smoke_dt (embed_dim 1024, tp=4,
# synthetic ERA5 subset -- see tests/make_synthetic_era5.py).
#
#   cd ~/tp-test-clone && qsub tests/submit_sunspot_train_tp.sh
#PBS -N vit-train-tp4
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

export CCL_LOG_LEVEL=warn
unset ZE_AFFINITY_MASK

python3 -c "import torch, sys; print('torch', torch.__version__); sys.exit(0 if torch.xpu.is_available() else 1)" || {
    echo "torch.xpu not available on this node"; exit 1;
}

# --- synthetic ERA5 subset (idempotent) --------------------------------------
if [ ! -f data/synthetic/train/data_2000.h5 ]; then
    python3 tests/make_synthetic_era5.py --out ./data/synthetic || exit 1
fi
ls -lh data/synthetic/train data/synthetic/valid data/synthetic/stats

echo "================ train.py --config=mp_smoke_dt --tensor_parallel=4 ================"
torchrun --standalone --nproc_per_node=4 \
    train.py --yaml_config=./config/ViT.yaml --config=mp_smoke_dt \
             --tensor_parallel=4 --device=xpu --run_num=tp4 || exit 1

echo "SUNSPOT TRAIN TP4: DONE"
