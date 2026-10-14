#!/bin/bash --login
# SPDX-License-Identifier: MIT
#
# PBS job: same smoke-training run as tests/submit_sunspot_train_tp.sh,
# but through the ezpz-based variant of train.py: ezpz provides the
# launcher (`ezpz launch`) and the in-process distributed/device setup
# (train.py --ezpz -> ezpz.distributed.setup_torch).
#
#   cd ~/tp-test-clone && qsub tests/submit_sunspot_train_tp_ezpz.sh
#PBS -N vit-train-tp4-ezpz
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

# a few of train.py's runtime dependencies are not in the frameworks env
# (tensorboard for SummaryWriter, ruamel.yaml for utils/YParams); install
# whatever is missing into the local venv, via the ALCF proxy
NEED_PIP=()
python3 -c "import tensorboard" 2>/dev/null || NEED_PIP+=(tensorboard)
python3 -c "import ruamel.yaml" 2>/dev/null || NEED_PIP+=(ruamel.yaml)
# NOTE: ezpz here is saforem2/ezpz (distributed-training utilities). The
# PyPI package named "ezpz" is an unrelated image tool -- install from git,
# and verify the import is the right package (it has ezpz.distributed).
python3 -c "import ezpz.distributed" 2>/dev/null || \
    NEED_PIP+=("git+https://github.com/saforem2/ezpz")
if [ "${#NEED_PIP[@]}" -gt 0 ]; then
    export http_proxy="${http_proxy:-http://proxy.alcf.anl.gov:3128}"
    export https_proxy="${https_proxy:-http://proxy.alcf.anl.gov:3128}"
    export HTTP_PROXY="${HTTP_PROXY:-$http_proxy}"
    export HTTPS_PROXY="${HTTPS_PROXY:-$https_proxy}"
    echo "pip-installing into venv: ${NEED_PIP[*]}"
    python3 -m pip install -q "${NEED_PIP[@]}" || exit 1
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

echo "================ ezpz launch: train.py --ezpz --config=mp_smoke_dt --tensor_parallel=4 ================"
ezpz launch --nproc 4 \
    python3 train.py --yaml_config=./config/ViT.yaml --config=mp_smoke_dt \
             --tensor_parallel=4 --ezpz --run_num=tp4ezpz || exit 1

echo "SUNSPOT TRAIN TP4 (ezpz): DONE"
