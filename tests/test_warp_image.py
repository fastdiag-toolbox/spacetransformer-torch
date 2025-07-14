import numpy as np
import torch
import pytest

from spacetransformer.core.space import Space
from spacetransformer.torch import warp_image, warp_image_batch


def random_space(shape=(10, 12, 14)):
    """生成默认 (x,y,z) 轴向, spacing=1 的 Space。"""
    return Space(shape=shape)


def test_identity_numpy():
    sp = random_space((6, 7, 8))
    img = np.random.rand(*sp.shape).astype(np.float32)
    out = warp_image(img, sp, sp, pad_value=0.0, cuda_device="cpu")
    assert np.allclose(out, img)


def test_crop_shift():
    src = random_space((10, 12, 14))
    # 裁剪 bbox (左闭右开)
    tgt = src.apply_bbox(np.array([[2, 8], [1, 10], [3, 13]]))
    img = np.random.rand(*src.shape).astype(np.float32)
    out = warp_image(img, src, tgt, pad_value=-1.0, cuda_device="cpu")
    gt = img[2:8, 1:10, 3:13]
    assert np.allclose(out, gt)


def test_flip_permute():
    src = random_space((5, 6, 7))
    tgt = src.apply_flip(0).apply_flip(2).apply_swap(0, 1)  # 复杂 flip + permute
    img = np.arange(np.prod(src.shape), dtype=np.float32).reshape(src.shape)
    out = warp_image(img, src, tgt, pad_value=0.0, cuda_device="cpu")
    # 手工 ground-truth
    gt = img[::-1, :, ::-1].transpose(1, 0, 2)
    assert np.allclose(out, gt)


def test_no_overlap():
    src = random_space((4, 4, 4))
    # 将目标完全移出范围
    tgt = src.apply_bbox(np.array([[10, 14], [10, 14], [10, 14]]))
    img = np.random.rand(*src.shape).astype(np.float32)
    pad_val = -5.0
    out = warp_image(img, src, tgt, pad_value=pad_val, cuda_device="cpu")
    assert np.all(out == pad_val)


def test_warp_image_batch():
    src = random_space((10, 10, 10))
    img = torch.rand(src.shape)
    targets = [
        src.apply_bbox(np.array([[0, 5], [0, 5], [0, 5]])).apply_shape((4, 4, 4)),
        src.apply_flip(1),
    ]
    outs = warp_image_batch(img, src, targets, pad_value=0.0, cuda_device="cpu")
    assert len(outs) == 2
    # 与单次调用一致
    for o, t in zip(outs, targets):
        single = warp_image(img, src, t, pad_value=0.0, cuda_device="cpu")
        assert torch.allclose(o, single) 