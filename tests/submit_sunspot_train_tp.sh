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

# tensorboard is the one runtime dependency the frameworks env lacks;
# install it into the local venv (via the ALCF proxy) only if missing
if ! python3 -c "import tensorboard" 2>/dev/null; then
    export http_proxy="${http_proxy:-http://proxy.alcf.anl.gov:3128}"
    export https_proxy="${https_proxy:-http://proxy.alcf.anl.gov:3128}"
    export HTTP_PROXY="${HTTP_PROXY:-$http_proxy}"
    export HTTPS_PROXY="${HTTPS_PROXY:-$https_proxy}"
    python3 -m pip install -q tensorboard || exit 1
fi

export CCL_LOG_LEVEL=warn
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
unset ZE_AFFINITY_MASK

python3 -c "import torch, sys; print('torch', torch.__version__); sys.exit(0 if torch.xpu.is_available() else 1)" || {
    echo "torch.xpu not available on this node"; exit 1;
}

# --- synthetic ERA5 subset (idempotent) --------------------------------------
if [ ! -f data/synthetic/train/data_2000.h5 ]; then
    python3 tests/make_synthetic_era5.py --out ./data/synthetic || exit 1
fi
ls -lh data/synthetic/train data/synthetic/valid data/synthetic/stats

# NOTE: `python3 -m torch.distributed.run`, not the `torchrun` script --
# the frameworks module's torchrun shebang points at the module python
# and would bypass this venv (and its extra packages) entirely.
echo "================ train.py --config=mp_smoke_dt --tensor_parallel=4 ================"
python3 -m torch.distributed.run --standalone --nproc_per_node=4 \
    train.py --yaml_config=./config/ViT.yaml --config=mp_smoke_dt \
             --tensor_parallel=4 --device=xpu --run_num=tp4 || exit 1

echo "SUNSPOT TRAIN TP4: DONE"
