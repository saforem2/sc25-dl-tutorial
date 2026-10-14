"""Generate a tiny synthetic dataset in the ERA5 layout train.py expects.

The real tutorial data is ERA5 .h5 files (one per year, dataset ``fields``
of shape [samples, channels, H, W]) plus global mean/std .npy stats. This
writes random-but-deterministic stand-ins so `train.py` can be smoke-tested
anywhere (e.g. Aurora) without the multi-GB download:

    python3 tests/make_synthetic_era5.py --out ./data/synthetic

Layout produced:
    <out>/train/data_2000.h5     fields: [n_train + 1, C, H, W] float32
    <out>/valid/data_2001.h5     fields: [n_valid + 1, C, H, W] float32
    <out>/stats/global_means.npy (shape (1, C, 1, 1), zeros)
    <out>/stats/global_stds.npy  (shape (1, C, 1, 1), ones)
    <out>/stats/time_means.npy   (same as global_means; unused by the
                                  pytorch loader but part of the layout)

The +1 sample is because ERA5Dataset pairs sample i (input) with i + dt
(target) inside the same year file.
"""

import argparse
import os

import h5py
import numpy as np


def write_year(path, n_samples, channels, img_x, img_y, rng):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with h5py.File(path, "w") as f:
        fields = f.create_dataset(
            "fields", shape=(n_samples + 1, channels, img_x, img_y),
            dtype="float32",
        )
        for i in range(n_samples + 1):
            # smooth-ish random field: low-res noise upsampled by repetition
            low = rng.standard_normal((channels, img_x // 8, img_y // 8))
            fields[i] = np.repeat(np.repeat(low, 8, axis=1), 8, axis=2)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="./data/synthetic")
    p.add_argument("--channels", type=int, default=20)
    p.add_argument("--img-x", type=int, default=192)
    p.add_argument("--img-y", type=int, default=384)
    p.add_argument("--n-train", type=int, default=64)
    p.add_argument("--n-valid", type=int, default=32)
    p.add_argument("--seed", type=int, default=1234)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    write_year(os.path.join(args.out, "train", "data_2000.h5"),
               args.n_train, args.channels, args.img_x, args.img_y, rng)
    write_year(os.path.join(args.out, "valid", "data_2001.h5"),
               args.n_valid, args.channels, args.img_x, args.img_y, rng)

    stats = os.path.join(args.out, "stats")
    os.makedirs(stats, exist_ok=True)
    shape = (1, args.channels, 1, 1)
    np.save(os.path.join(stats, "global_means.npy"), np.zeros(shape, "float32"))
    np.save(os.path.join(stats, "time_means.npy"), np.zeros(shape, "float32"))
    np.save(os.path.join(stats, "global_stds.npy"), np.ones(shape, "float32"))
    print(f"wrote synthetic ERA5 data under {args.out}")


if __name__ == "__main__":
    main()
