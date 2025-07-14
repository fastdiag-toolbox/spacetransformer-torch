import numpy as np
import torch

from spacetransformer.core import Space
from spacetransformer.torch import warp_image


def test_crop_warp():
    """裁剪 + warp 验证像素级一致。"""
    src = Space(shape=(10, 12, 14))
    tgt = src.apply_bbox(np.array([[2, 8], [3, 11], [1, 10]]))

    img = np.random.rand(*src.shape).astype(np.float32)
    out = warp_image(img, src, tgt, pad_value=0.0, cuda_device="cpu")
    gt = img[2:8, 3:11, 1:10]
    assert np.allclose(out, gt)


def test_2d_slice_case():
    """当某轴长度为 1 时应退化为 2D，不触发 grid_sample bug。"""
    sp1 = Space(shape=(20, 20, 1))  # 第三轴为 1
    # bbox 后再缩放到 (6,6,1)
    sp2 = sp1.apply_bbox(np.array([[5, 16], [5, 16], [0, 1]])).apply_shape((6, 6, 1))

    img = torch.rand(sp1.shape)
    out = warp_image(img, sp1, sp2, pad_value=0.0, cuda_device="cpu", mode="nearest")
    gt = img[5:16:2, 5:16:2, :]
    assert out.shape == gt.shape 