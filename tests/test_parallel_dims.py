"""Unit tests for the TP/FSDP2 parallel-degree bookkeeping.

Pure bookkeeping -- no process group needed:

    python -m pytest tests/test_parallel_dims.py
    # or
    python -m unittest tests.test_parallel_dims
"""

import unittest

from utils.parallel_dims import resolve_dp_degrees, validate_tp_for_vit


class TestResolveDpDegrees(unittest.TestCase):
    def test_flat_dp_default(self):
        # tp=2 on 8 ranks -> single sharded dp dim of 4
        self.assertEqual(resolve_dp_degrees(8, tp=2), (1, 4))

    def test_tp1(self):
        self.assertEqual(resolve_dp_degrees(4, tp=1), (1, 4))

    def test_all_tp(self):
        self.assertEqual(resolve_dp_degrees(4, tp=4), (1, 1))

    def test_hsdp_auto_shard(self):
        # dp_replicate=2 on 16 ranks with tp=2 -> dp_shard = 16 / (2*2) = 4
        self.assertEqual(
            resolve_dp_degrees(16, tp=2, dp_replicate=2, dp_shard=-1), (2, 4)
        )

    def test_hsdp_explicit(self):
        self.assertEqual(
            resolve_dp_degrees(16, tp=2, dp_replicate=2, dp_shard=4), (2, 4)
        )

    def test_indivisible_world(self):
        with self.assertRaises(AssertionError):
            resolve_dp_degrees(6, tp=4)

    def test_product_mismatch(self):
        with self.assertRaises(AssertionError):
            resolve_dp_degrees(16, tp=2, dp_replicate=2, dp_shard=3)

    def test_bad_degrees(self):
        with self.assertRaises(AssertionError):
            resolve_dp_degrees(8, tp=0)
        with self.assertRaises(AssertionError):
            resolve_dp_degrees(8, tp=2, dp_replicate=0)
        with self.assertRaises(AssertionError):
            resolve_dp_degrees(8, tp=2, dp_shard=0)


class TestValidateTpForVit(unittest.TestCase):
    # the `base` config from config/ViT.yaml
    BASE = dict(embed_dim=384, num_heads=8, mlp_ratio=4, patch_size=8, out_chans=20)
    # the `mp` config: bigger embed_dim, same heads/patch
    MP = dict(BASE, embed_dim=1024)

    def test_base_config(self):
        for tp in (1, 2, 4, 8):
            validate_tp_for_vit(tp=tp, **self.BASE)

    def test_mp_config(self):
        for tp in (2, 4, 8):
            validate_tp_for_vit(tp=tp, **self.MP)

    def test_tp_must_divide_heads(self):
        with self.assertRaises(AssertionError):
            validate_tp_for_vit(tp=3, **self.BASE)

    def test_tp_larger_than_heads(self):
        with self.assertRaises(AssertionError):
            validate_tp_for_vit(tp=16, **self.BASE)


if __name__ == "__main__":
    unittest.main()
